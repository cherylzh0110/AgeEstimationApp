import cv2
import json
import urllib.request
from urllib.request import Request, urlopen
import base64
import PIL
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response, jsonify
import logging
import os
import io
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from keras.applications import imagenet_utils
from utkcamera import VideoCamera
from model import get_model

app = Flask(__name__,static_url_path='')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img_width, img_height = 119, 119
target_size=(img_width,img_height)

model = load_model('./models/model64.h5')
model._make_predict_function()
#model1 = load_model('./models/model_weights.h5')
#model1._make_predict_function()

def preprocess_img(img,target_size=(119,119)):
    if (img.shape[2] == 4):
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)    
    img = cv2.resize(img,target_size)
    img = np.divide(img,255.)
    img = np.subtract(img,0.5)
    img = np.multiply(img,2.)
    return img

def preprocess_image(filename, target_size):
    #if image.mode != "RGB":
        #image = image.convert("RGB")
    image = cv2.imread(filename)
    #image = np.array(image, "float32")
    faces = face_cascade.detectMultiScale(image,
                                          scaleFactor=1.1,
                                          minNeighbors=3, )
    largesize = 0
    for (x, y, w, h) in faces:
        roi = image[y:y + h, x:x + w]
        size = roi.size
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
        if size > largesize:
            largesize = size
            finalroi = roi
    filename = './crop.png'
    cv2.imwrite("crop.png", finalroi)
    image = Image.open(filename)
    image = image.resize(target_size, Image.ANTIALIAS)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def load_im_from_system(url):
    image_url = url.split(',')[1]
    image_url = image_url.replace(" ", "+")
    image_array = base64.b64decode(image_url)
    image = Image.open(io.BytesIO(image_array))
    print(image)# wrap the bytes and pass to open image
    image.save("new.png")
    processed_image = preprocess_image("./new.png", target_size=(119, 119))
        #(1, 119, 119, 3)
    return processed_image

def togroupname(i):
    Y_label = ""
    if i == 0:
        Y_label = "From 0-2"
    elif i == 1:
        Y_label = "From 3-6"
    elif i == 2:
        Y_label = "From 7-14"
    elif i == 3:
        Y_label = "From 15-23"
    elif i == 4:
        Y_label = "From 24-34"
    elif i == 5:
        Y_label = "From 35-44"
    elif i == 6:
        Y_label = "From 45-59"
    elif i == 7:
        Y_label = "Above 60"
    return Y_label

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/classify_system', methods=['GET'])
def classify_system():
    #send get request to server to get url and return predict result to server
    #in the server side, it use $.get method to execute GET requests from client in the classifysystem function we define and receive data
    image_url = request.args.get('imageurl')
    image_array = load_im_from_system(image_url)
    image_array = image_array.astype('float32')
    image_array /= 255
    print(image_array.shape)  #(1, 32, 32, 3)
    prediction = model.predict(image_array)
    #prediction1 = model1.predict(image_array)
    #print(prediction)  # 2-DIM PROBABILITY ARRAY
    # [[9.96262848e-01 3.45902634e-03 2.38975495e-04 1.83143529e-05 4.15922977e-06 1.18151775e-05 3.91400044e-06 1.09522546e-06]]
    #print(prediction[0])  # AN ARRAY OF PROBABILITY
    a = prediction[0].tolist()
    print(type(prediction[0]))  # NP.NDARRAY
    id = prediction[0].argmax()
    text = "Label" + str(id)
    print(text)
    result = []
    for r in prediction[0]:
        result.append({"name": togroupname(a.index(r)), "y": float(r) * 100})
        # "{0:.0f}%".format(float(r) * 100)
        # TRUN ARRAY INTO LIST

    #ages = np.arange(0, 101).reshape(101, 1)
    #predicted_ages = prediction1.dot(ages).flatten()
    #finalscore = predicted_ages[0]
    #print(finalscore)
    print(result)
    return jsonify({'results':result})

@app.route('/classify-system', methods=['GET'])
def do_system():
    return app.send_static_file('system.html')

@app.route('/', methods=['GET'])
def root():
    return app.send_static_file('index.html')

@app.route('/about', methods=['GET'])
def about():
    return app.send_static_file('about.html')

if __name__ == '__main__':
    app.run(debug=True)

