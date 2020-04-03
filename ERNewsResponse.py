#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 15:17:48 2020

@author: jimmyjacobson
"""
import requests
import pprint
import random
import pandas as pd
import nltk
import string

def ER_APIRequest():
    eventregistry_api_key = '914fb9d8-0fee-4699-8670-41057659bc4d'
    
    # Define the news api endpoint
    url = 'http://eventregistry.org/api/v1/article/getArticles'
    
    # Specify the query and number of returns
    parameters = {
    "action": "getArticles",
    "lang": "eng",
    "keyword": "covid-19",
    "articlesPage": 1,
    "articlesCount": 100,
    "articlesSortBy": "socialScore",
    "startSourceRankPercentile": 0,
    "endSourceRankPercentile": 30,
    "includeArticleSocialScore": True,
    "includeSourceRanking" : True,
    "includeConceptTrendingScore" : True,
    "includeArticleConcepts" : True,
    "articlesSortByAsc": False,
    "articlesArticleBodyLen": -1,
    "resultType": "articles",
    "dataType": "news",
    "apiKey": eventregistry_api_key,
    "forceMaxDataTimeWindow": 31,
}
    
    # Make the request
    response = requests.get(url, params=parameters)
    
    # Convert the response to JSON format
    response_json = response.json()
    #pprint.pprint(response_json)
    
    # for i in response_json['articles']:
    #     print(i['title'])
    #     print(i['publishedAt'])
    #     print(i['source']['name'])
    #     print('\n')
    
    return response_json

mResponse = ER_APIRequest()
    
#print(mResponse['articles'][1])
#mResponse['articles']

#Prepopulate sentiment score with zeros
for i in range(len(mResponse['articles']['results'])):
    publisher = mResponse['articles']['results'][i]['source']['title']
    mResponse['articles']['results'][i].update({'sentiment_score' : random.random(), 'relevance_score' : random.random(), 'publisher' : publisher})

# # Create pandas df
Mdf = pd.DataFrame.from_dict(mResponse['articles']['results'])

## Unigram feature extraction

# Get brown news words
from nltk.corpus import brown
s = brown.words(categories='news')


stopwords = nltk.corpus.stopwords.words('english')
all_words = [w for w in s if w.lower() not in stopwords] 
x = [''.join(c for c in s if c not in string.punctuation) for s in all_words]
x = [s for s in x if s]
words = nltk.FreqDist(w.lower() for w in x)
word_features = list(words)[:2000]
    
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
path = get_tmpfile("word2vec.model")
model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")
    ## WORKSHOP IDEAS HERE WITH THE TAGGED SENTENCES ###
    
    
    
    
    # adjs = dict()
    # for i in range(len(tagged)):
    #     tag = tagged[i][1]
    #     if tag == 'JJ':
    #         #print(tagged[i])
    #         if tagged[i][0] not in adjs:
    #             adjs.update({tagged[i][0] : 1})
    #         else:
    #             adjs[tagged[i][0]] += 1
    # print(adjs)
        #mResponse['articles']['results'][i].update

# import plotly.express as px
# fig = px.scatter_3d(Mdf, x='publishedAt', y='relevance_score', z='sentiment_score',
#                     color='publisher', symbol='publisher')
# plotly.offline.plot(fig, filename='3d.html')