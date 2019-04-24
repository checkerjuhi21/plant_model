from flask import Flask, request, jsonify
import os
import numpy as np
import cv2
from imageai.Prediction.Custom import CustomImagePrediction

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__)
image1 = None

prediction = CustomImagePrediction()

prediction.setModelTypeAsInceptionV3()

prediction.setModelPath("model_ex-053_acc-0.997352.h5")

prediction.setJsonPath("model_class.json")



def model_stra_pota():
    
    prediction.loadModel(num_objects=31)
    predictions, probabilities = prediction.predictImage('img.jpg', result_count=1)
    print (predictions, probabilities)
    predictions = predictions[0]
    probabilities = probabilities[0]
    ans={"pred":predictions,"prob":probabilities}

    #e = {predictions:probabilities}
    #print (e)
    #for eachPrediction, eachProbability in zip(predictions, probabilities):
    #    print(eachPrediction + " : " + eachProbability)
    return (ans)

#endpoint to predict disease
@app.route("/predict", methods=["POST"])
def add_user():
    file    = request.files['image'].read()
    npimg   = np.fromstring(file,np.uint8)
    img     = cv2.imdecode(npimg,1)
    cv2.imwrite('img.jpg',img )
    ans     = model_stra_pota()
    return jsonify(ans)

@app.route('/', methods=['GET'])
def home():
    print("loaded")
    return "Welcome to My API"


if __name__ == '__main__':
    app.run()
