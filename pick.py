# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 00:38:14 2021

@author: Debayan
"""
import pickle

class extract:
    def word_to_chars(word):
        characters = []
        for char in word:
            characters.append(char)
        return characters 
    
    def tfidf_vectorizer(self):
        appfile1 = open("vectorizer.pkl","rb")
        vectorizer = pickle.load(appfile1)
        return vectorizer
    
    def xgboost_model(self):
        appfile = open("PassCheck_XGBoost.pkl","rb")
        model = pickle.load(appfile)
        return model
