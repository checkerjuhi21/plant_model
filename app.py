from flask import Flask, request, jsonify
import os
import numpy as np
import cv2
from imageai.Prediction.Custom import CustomImagePrediction

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__)

image1 = None

def model_stra_pota(img):
    prediction = CustomImagePrediction()

    prediction.setModelTypeAsInceptionV3()

    prediction.setModelPath("epoch_44th.h5")

    prediction.setJsonPath("model_class.json")

    prediction.loadModel(num_objects=5)

    predictions, probabilities = prediction.predictImage(img, result_count=1,input_type="array")
    print (predictions, probabilities)
    predictions = predictions[0]
    probabilities = probabilities[0]
    e = {predictions:probabilities}
    #print (e)
    #for eachPrediction, eachProbability in zip(predictions, probabilities):
    #    print(eachPrediction + " : " + eachProbability)

    return (e)

#endpoint to predict disease 
@app.route("/predict", methods=["POST"])
def add_user():
    file    = request.files['image'].read() 
    npimg   = np.fromstring(file,np.uint8)
    img     = cv2.imdecode(npimg,1) 
    ans     = model_stra_pota(img)     
    return jsonify(ans)


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
    app.run()
