"""
Created on Fri Jun 21 11:03:07 2024

@author: Johan Agoualè Kouamé (DS@ST2I) 
"""
import os 
from flask import Flask,request,app,jsonify,url_for,render_template
import pickle 
import json 
import numpy as np
import pandas as pd
import seaborn as sns

app=Flask(__name__)
##Load the model 

regmodel=pickle.load(open('regrf.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0]) 

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House value prediction is {}".format(output))


if __name__=="__main__":
    app.run(debug=True)
   