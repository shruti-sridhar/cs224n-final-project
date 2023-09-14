# CS 224N Final Project - A Sentence-BERT Extension to the minBERT Model

This is our code for the default final project for the Stanford CS 224N class. Our final report is [here](https://web.stanford.edu/class/cs224n/final-reports/final-report-169956358.pdf).

In this project, we aim to fine-tune the minBERT model to simultaneously perform well on sentiment analysis, paraphrase detection, and semantic textual similarity (STS) prediction tasks. First, we use pre-trained weights loaded into our minBERT implementation and train only for the sentiment task to obtain baseline performance metrics for all three downstream tasks. Second, we train for all three tasks at once, using multi-task finetuning and gradient surgery to finetune our embeddings. We take an approach inspired by Sentence-BERT (SBERT) to generate embeddings that can be compared via cosine similarity for the STS task, addressing the overhead of computing pairwise similarities with BERT. Overall, our finetuned embeddings outperform our baseline on two out of the three tasks.

### Acknowledgement

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html), created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig. 

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
