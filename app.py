#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:10:39 2019

@author: juhichecker
"""

from flask import Flask, request, jsonify
#from flask_marshmallow import Marshmallow
import os
from imageai.Prediction.Custom import CustomImagePrediction

UPLOAD_FOLDER = '/Users/juhichecker/Desktop/Deep'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



image1 = None

def model_stra_pota():

    execution_path = '/Users/juhichecker/Deep'
    test_path = '/Users/juhichecker/Deep/'

    prediction = CustomImagePrediction()

    prediction.setModelTypeAsInceptionV3()

    prediction.setModelPath(os.path.join(execution_path, "epoch_44th.h5"))

    prediction.setJsonPath(os.path.join(execution_path, "model_class.json"))

    prediction.loadModel(num_objects=5)

    predictions, probabilities = prediction.predictImage(os.path.join('/Users/juhichecker/Deep/', 'image.jpg'), result_count=1)
    print (predictions, probabilities)
    predictions = predictions[0]
    probabilities = probabilities[0]
    e = {predictions:probabilities}
    #print (e)
    #for eachPrediction, eachProbability in zip(predictions, probabilities):
    #    print(eachPrediction + " : " + eachProbability)

    return (e)



# endpoint to create new user
@app.route("/user", methods=["POST"])
def add_user():
    file = request.files['image']
    file.save("/Users/juhichecker/Deep/image.jpg")
    
    return "sd"


# endpoint to show all users
@app.route("/user", methods=["GET"])
def get_user():

    e = model_stra_pota()
    return jsonify(e)

@app.route('/', methods=['GET'])
def home():
    print("loaded")
return "Welcome to My API"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5002)
