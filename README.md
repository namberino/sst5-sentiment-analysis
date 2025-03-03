# Fine-grained Sentiment Analysis

This project aims to perform sentiment analysis on the [SST-5](https://paperswithcode.com/dataset/sst-5) dataset, which contains 5 labels. This is a fine-grained sentiment analysis task. The highest recorded accuracy on this dataset to date is 59.8%, which was achieved by Franz A. Heinsen in the [An Algorithm for Routing Vectors in Sequences](https://arxiv.org/pdf/2211.11754v3) paper using Heinsen Routing + RoBERTa Large. A leaderboard for the SST-5 dataset can be found [here](https://paperswithcode.com/sota/sentiment-analysis-on-sst-5-fine-grained).

This notebook aims to compare some sentiment analysis methods on this dataset. Here are the methods we tested along with the recorded accuracy for each of the model:
- Embedding + CNN: 40%
- Multinomial Naive Bayes: 38.51%
- Logistic Regression: 40.63%
- BERT Base: 53.57%
- BERT Large: 54.93%
