# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:48:30 2021

@author: Debayan
"""

import pickle
from flask import Flask, request
import numpy as np
from pick import extract


app1 = Flask(__name__)
strength = ["weak", "medium", "strong"]

@app1.route('/', methods=['GET'])
def start():
    return "Check password"

@app1.route('/predict', methods=['GET'])
def checkPass():
     ext = extract()
     model = ext.xgboost_model()
     vectorizer = tfidf_vectorizer()
     password = request.args.get("pass")
     X_predict=np.array([password])
     X_predict=vectorizer.transform(X_predict)
     value = model.predict(X_predict)[0]
     return "The strength of password is {}".format(strength[value])
   
if __name__=='__main__':
    app1.run()


# xgboost-1.4.2

