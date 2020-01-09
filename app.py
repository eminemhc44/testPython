#import sys
import os
#import glob
#import re
import numpy as np
import cv2


from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from keras.models import load_model


    

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
model_path = 'models/rps_model.h5'
weight_path = 'models/rps_weight-improvement-08-0.83.hdf5'

# Load your trained model
model = load_model(model_path)
model.load_weights(weight_path)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
print('Model loaded. Check http://127.0.0.1:5000/')



def model_predict(img_path, model):
    img = cv2.imread(img_path)

    # Preprocessing the image
    img = img/255.0
    img = cv2.resize(img,(224,224))
    img = np.expand_dims(img,axis=0)

    preds = np.argmax(model.predict(img))

    class_dict = {0:'paper',1:'rock',2:'scissors'}

    result = class_dict[preds]
    
    return result





@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'user_uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
#        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(preds)               # Convert to string
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()



























