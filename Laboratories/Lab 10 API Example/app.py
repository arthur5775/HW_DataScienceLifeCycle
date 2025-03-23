from flask import Flask, request
import numpy as np
import pickle
import ast

app = Flask(__name__)

def predict_diabetes(name):
    if name == False or type(name) != str:
        return "Name must contain letters and be of type string"
    classifier = pickle.load(open('data/final_model.sav', 'rb'))
    X = np.array(ast.literal_eval(name)).reshape(1,-1)
    y_pred = classifier.predict(X)
    if y_pred == [1.0]:
        return "Positive, patient is diabetic"
    else:
        return "Negative, patient is not diabetic"

# This is a HTTP method, you can read about it on your own

@app.route('/predict', methods=['GET'])

def predict():
    return predict_diabetes(str(request.query_string, 'utf-8'))