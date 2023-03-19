import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace
from enum import Enum
from typing import Iterable, Dict
from itertools import zip_longest

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sentence_transformers
from sentence_transformers import losses

import pcgrad
from pcgrad import PCGrad

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, model_eval_multitask, test_model_multitask

TQDM_DISABLE=True

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        self.dropout = torch.nn.Dropout(p=config.hidden_dropout_prob)
        self.sent_linear = torch.nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.para_linear_cat = torch.nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        self.para_linear_dist = torch.nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
        self.sts_linear = torch.nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).

        embedding_output = self.bert.embed(input_ids=input_ids)
        sequence_output = self.bert.encode(embedding_output, attention_mask=attention_mask)

        # get cls token hidden state
        first_tk = sequence_output[:, 0]
        first_tk = self.bert.pooler_dense(first_tk)
        first_tk = self.bert.pooler_af(first_tk)

        return first_tk


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''

        embedding = self.forward(input_ids, attention_mask)
        embedding = self.dropout(embedding)
        logits = self.sent_linear(embedding)
        
        return logits


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        
        embedding_1 = self.forward(input_ids_1, attention_mask_1)
        embedding_1 = self.dropout(embedding_1)
        embedding_2 = self.forward(input_ids_2, attention_mask_2)
        embedding_2 = self.dropout(embedding_2)

        if args.loss_type_para == "cosine":
            embedding_1 = self.para_linear_dist(embedding_1)
            embedding_2 = self.para_linear_dist(embedding_2)
            similarity = (F.cosine_similarity(embedding_1, embedding_2)).float()
            logit = (similarity + 1) / 2 # scale to [0, 1] range

        if args.loss_type_para == "BCE":
            both_embeddings = torch.cat((embedding_1, embedding_2), dim=1)
            logit = self.para_linear_cat(both_embeddings).squeeze()

        return logit


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        
        embedding_1 = self.forward(input_ids_1, attention_mask_1)
        embedding_1 = self.sts_linear(embedding_1)

        embedding_2 = self.forward(input_ids_2, attention_mask_2)
        embedding_2 = self.sts_linear(embedding_2)

        if args.loss_type_sts == "MSE":
            logit = (F.cosine_similarity(embedding_1, embedding_2)).float() # in [-1, 1] range

        if args.loss_type_sts == "cosine":
            logit = (F.cosine_similarity(embedding_1, embedding_2)).float() # in [-1, 1] range
            logit = (logit + 1) / 2 # scale to [0, 1] range to be compatible with CosineEmbeddingLoss
        
        return logit


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer,
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_batch_size = 32
    para_batch_size = 32
    sts_batch_size = 32

    # 1. SENTIMENT CLASSIFICATION TRAIN/DEV DATA (SST)
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=False, batch_size=sst_batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=sst_batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    # 2. PARAPHRASE DETECTION TRAIN/DEV DATA (QUORA)
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args) 

    para_train_dataloader = DataLoader(para_train_data, shuffle=False, batch_size=para_batch_size,
                                          collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=para_batch_size,
                                         collate_fn=para_dev_data.collate_fn)


    # 3. SEMANTIC TEXTUAL SIMILARITY TRAIN/DEV DATA (SEMEVAL)
    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)
    
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=False, batch_size=sts_batch_size,
                                         collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=sts_batch_size,
                                        collate_fn=sts_dev_data.collate_fn)
    
    # Initialize model configuration
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    if args.load_saved_model:
        saved = torch.load(args.filepath)
        config = saved['model_config']
        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model for further training from {args.filepath}")
    else:
        model = MultitaskBERT(config)
        model = model.to(device)

    lr = args.lr
    
    if args.use_grad_surgery:
        optimizer = PCGrad(AdamW(model.parameters(), lr=lr))
    else:
        optimizer = AdamW(model.parameters(), lr=lr)

    best_dev_acc_sst = 0
    best_dev_acc_para = 0
    best_dev_corr_sts = 0

    # Run training loop for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        print("\nRunning Epoch {}...".format(epoch))
        print("===================")

        generator_sst = iter(sst_train_dataloader)
        generator_para = iter(para_train_dataloader)
        generator_sts = iter(sts_train_dataloader)

        for i, (batch_sst, batch_para, batch_sts) in enumerate(zip_longest(sst_train_dataloader, para_train_dataloader, sts_train_dataloader)):
            
            # STEP 1. train on one batch from sentiment SST dataset
            try:
                batch_sst = next(generator_sst)
            except StopIteration:
                generator_sst = iter(sst_train_dataloader)
                batch_sst = next(generator_sst)

            if batch_sst:
                b_ids, b_mask, b_labels = (batch_sst['token_ids'],
                                           batch_sst['attention_mask'], batch_sst['labels'])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                logits = model.predict_sentiment(b_ids, b_mask)
                loss_sentiment = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / sst_batch_size
                loss_sentiment = loss_sentiment.to(device)


            # STEP 2. train on one batch from paraphrase Quora dataset
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch_para['token_ids_1'], batch_para['attention_mask_1'],
                          batch_para['token_ids_2'], batch_para['attention_mask_2'],
                          batch_para['labels'], batch_para['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels = b_labels.to(device)
            b_labels = b_labels.type(torch.FloatTensor)

            logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            logits = logits.type(torch.FloatTensor)

            if args.loss_type_para == "cosine":
                # logits represent cosine distances: maximize/minimize distance based on true label
                loss_paraphrase = 0.5 * (b_labels.float() * logits.pow(2) + (1 - b_labels).float() * F.relu(0.5 - logits).pow(2))
                loss_paraphrase = loss_paraphrase.sum() / para_batch_size

            if args.loss_type_para == "BCE":
                # logits normalized to be probabilities from 0-1: compute BCE loss with true labels
                logits = torch.sigmoid(logits)
                loss_paraphrase = F.binary_cross_entropy(logits, b_labels.view(-1), reduction='sum') / para_batch_size

            loss_paraphrase = loss_paraphrase.to(device) 

            # STEP 3. train on one batch from semantic textual similarity SemEval dataset
            try:
                batch_sts = next(generator_sts)
            except StopIteration:
                generator_sts = iter(sts_train_dataloader)
                batch_sts = next(generator_sts)

            if batch_sts:
                (b_ids1, b_mask1,
                 b_ids2, b_mask2,
                 b_labels, b_sent_ids) = (batch_sts['token_ids_1'], batch_sts['attention_mask_1'],
                              batch_sts['token_ids_2'], batch_sts['attention_mask_2'],
                              batch_sts['labels'], batch_sts['sent_ids'])

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)
                b_labels = b_labels.to(device)
                b_labels = b_labels.type(torch.FloatTensor)

                logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                logits = logits.type(torch.FloatTensor)

                # MSE loss: rescale STS labels from [0, 5] to [-1, 1] to be compatible with cosine similarities
                b_labels = 2 / 5 * (b_labels - 5) + 1
                
                loss_similarity = F.mse_loss(logits, b_labels.view(-1), reduction='sum') / sts_batch_size
                loss_similarity = loss_similarity.to(device)

            # STEP 4. backpropagate losses (with gradient surgery if applicable)
            optimizer.zero_grad()
            
            # if not batch_sst: loss_sentiment = 0
            # if not batch_sts: loss_similarity = 0
            
            total_loss = (loss_sentiment + loss_paraphrase + loss_similarity).float()
            
            # if not batch_sst and not batch_sts:
            #     losses = [loss_paraphrase.float()]
            # elif not batch_sts:
            #     losses = [loss_sentiment.float(), loss_paraphrase.float()]
            # else:
            
            losses = [loss_sentiment.float(), loss_paraphrase.float(), loss_similarity.float()]

            if args.use_grad_surgery:
                optimizer.pc_backward(losses)
            else:
                total_loss.backward()
            
            optimizer.step()

            train_loss += total_loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)
        
        # STEP 5. collect train and dev accuracies, save model if improved
        print("\ntrain accuracies...")
        train_paraphrase_accuracy, train_para_y_pred, train_para_sent_ids, \
            train_sentiment_accuracy, train_sst_y_pred, train_sst_sent_ids, train_sts_corr, \
            train_sts_y_pred, train_sts_sent_ids = model_eval_multitask(sst_train_dataloader, para_train_dataloader, 
                                                                                sts_train_dataloader, model, device)
        
        print("\ndev accuracies...")
        dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sentiment_accuracy, dev_sst_y_pred, dev_sst_sent_ids, dev_sts_corr, \
            dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, 
                                                                            sts_dev_dataloader, model, device)

        improved_model = False

        if dev_sentiment_accuracy > best_dev_acc_sst:
            best_dev_acc_sst = dev_sentiment_accuracy
            improved_model = True
        if dev_paraphrase_accuracy > best_dev_acc_para:
            best_dev_acc_para = dev_paraphrase_accuracy
            improved_model = True
        if dev_sts_corr > best_dev_corr_sts:
            best_dev_corr_sts = dev_sts_corr
            improved_model = True

        if improved_model: save_model(model, optimizer, args, config, args.filepath)

        print(f"\nEpoch {epoch}: train loss :: {train_loss :.3f}\n")


def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    # additional arguments
    parser.add_argument("--loss_type_para", type=str, help="loss type for paraphrase detection (BCE or cosine)", default="BCE")
    parser.add_argument("--loss_type_sts", type=str, help="loss type for STS evaluation (MSE or cosine)", default="MSE")
    parser.add_argument("--use_grad_surgery", help="use gradient surgery while multi-task finetuning", action='store_true')
    parser.add_argument("--load_saved_model", help="load an existing model for further training", action='store_true')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    grad_surgery = "surg" if args.use_grad_surgery else "no-surg"
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-{grad_surgery}-{args.loss_type_para}-{args.loss_type_sts}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
