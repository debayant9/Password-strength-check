# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 00:38:14 2021

@author: Debayan
"""
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer 


class extract:
    def word_to_chars(word):
        characters = []
        for char in word:
            characters.append(char)
        return characters 
    
    def tfidf_vectorizer(self):
        vectorizer = TfidfVectorizer(tokenizer=self.word_to_chars)
        return vectorizer
    
    def xgboost_model(self):
        appfile = open("PassCheck_XGBoost.pkl","rb")
        model = pickle.load(appfile)
        return model
