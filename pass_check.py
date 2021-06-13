# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:48:30 2021

@author: Debayan
"""

import pickle
from flask import Flask, request
import numpy as np
import pick


app1 = Flask(__name__)
strength = ["weak", "medium", "strong"]

appfile = open("PassCheck_XGBoost.pkl","rb")
model = pickle.load(appfile)



@app1.route('/', methods=['GET'])
def start():
    return "Check password"

@app1.route('/predict', methods=['GET'])
def checkPass():
    appfile1 = open("vectorizer.pkl","rb")
    vectorizer = pickle.load(appfile1)
    password = request.args.get("pass")
    X_predict=np.array([password])
    X_predict=vectorizer.transform(X_predict)
    value = model.predict(X_predict)[0]
    return "The strength of password is {}".format(strength[value])


if __name__=='__main__':
    from pick import word_to_chars
    app1.run()


# xgboost-1.4.2

