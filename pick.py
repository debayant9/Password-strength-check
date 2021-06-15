# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 00:38:14 2021

@author: Debayan
"""
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer 

def word_to_chars(word):
        characters = []
        for char in word:
            characters.append(char)
        return characters 

class extract:
    
    
    def tfidf_vectorizer(self):
        x_data_file = open("deploy_train_data.pkl","rb")
        x_data = pickle.load(x_data_file)
        vectorizer = TfidfVectorizer(tokenizer=word_to_chars)
        vectorizer.fit_transform(x_data)
        return vectorizer
    
    def xgboost_model(self):
        appfile = open("PassCheck_XGBoost.pkl","rb")
        model = pickle.load(appfile)
        return model
