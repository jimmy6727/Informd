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
import re
import os
from sklearn import svm
import array
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, GRU
from keras.layers.embeddings import Embedding

# Read data returns an array of vectorized sentence, label, longest
def ReadData(fname):
    print("Preprocessing data...")
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

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

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
t = ReadData('data/MT/002.train')
for i in range(len(t[0])):
    docs.append(TaggedDocument(t[0][i], [t[1][i]]))

test=t[0]

# Build doc2vec  model
model=Doc2Vec(vector_size=300, min_count=3, epochs=40)
model.build_vocab(docs)
model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)


print("Training...")
traindata=t
X_train=toVectorRep(traindata[0])
y_train=traindata[1]
testdata=ReadData('data/MT/002.test')
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

# Takes text
def IDClaims(classifier, text):
    sentences=split_into_sentences(text)
    print("1 indicates a claim, 0 indicates no claim\n")
    for i in range(len(sentences)):
        sentences[i]=nltk.word_tokenize(sentences[i])
    vecs=toVectorRep(sentences)
    predicted = classifier.predict(vecs)
    for i in range(len(predicted)):
        print("""Prediction for sentence""",str(i),""":\t\t""",str(predicted[i]).rjust(10))

