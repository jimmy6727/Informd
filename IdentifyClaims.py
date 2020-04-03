# -*- coding: utf-8 -*-

import sklearn
import pandas as pd
import numpy as np
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import nltk
from nltk.corpus import brown
import string
import smart_open
import os
from sklearn import svm
import array
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, GRU
from keras.layers.embeddings import Embedding

# Read data returns an array of vectorized sentence, label
def ReadData(fname):
    data=open(fname)
    sents=[]
    labels = []
    longest=0
    for line in data:
        sent = nltk.word_tokenize(line[:-3])
        new_l=len(sent)
        if new_l>longest:
            longest=new_l
        sents.append(sent)
        label = int(line[-2])
        labels.append(label)
    return [sents,labels,longest]

def toVectorRep(sentences):
    out=np.empty([len(sentences),300])
    for i in range(len(sentences)):
        a = model.infer_vector(sentences[i])
        out[i]=a
    return out

# Build corpus of news words
s = brown.words(categories='news')
stopwords = nltk.corpus.stopwords.words('english')
all_words = [w for w in s if w.lower() not in stopwords] 
x = [''.join(c for c in s if c not in string.punctuation) for s in all_words]
corpus = [s for s in x if s]
fd = nltk.FreqDist(w.lower() for w in corpus)

docs=[]
t = ReadData('data/MT/001.train')
for i in range(len(t[0])):
    docs.append(TaggedDocument(t[0][i], [t[1][i]]))

test=ReadData('data/MT/001.train')[0]

# Build doc2vec  model
model=Doc2Vec(vector_size=300, min_count=3, epochs=40)
model.build_vocab(docs)
model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)


print("Training...")
traindata=ReadData('data/MT/001.train')
X_train=toVectorRep(traindata[0])
y_train=traindata[1]
testdata=ReadData('data/MT/001.test')
X_test=toVectorRep(testdata[0])
y_test=testdata[1]

print("Testing...")
c1 = svm.SVC()
c1.fit(X_train,y_train)
predicted = c1.predict(X_test)
c,t=0,0
for i in range(len(X_test)):
    print("Predicted "+str(predicted[i])+"\t\t Actual "+str(y_test[i]))
    if predicted[i]==y_test[i]:
        c+=1
    t+=1
print("\nAccuracy: "+str(c/t)+"%")

