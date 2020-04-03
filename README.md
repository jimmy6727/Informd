# Informd
Project repo for Mozilla Spring Incubator Lab 2020 Project 

Included in the initial commit is a machine learning algorithm that uses a SVM classfier to identify whether or not a given sentence is a claim. Sentences are mapped to 50-dimensional vector embeddings to be fed as input into the classfier using gensim doc2vec, as used in Mikolov et al in https://arxiv.org/pdf/1405.4053v2.pdf. 

This algorithm will be improved upon and used in our automated analysis of news articles. Initial testing yielded 83% prediction accuracy.

Also included is the data that our model was trained with, and a script that gets top articles (Ranked by total number of shares on social media) from the EventRegistry news API.
