# camera.py contains a prediction of real-time image based on 8 classes model

import cv2
from keras.models import Sequential, load_model
import numpy as np

facec = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

def get_model():
    global model
    model = load_model('./models/model64.h5')
    model._make_predict_function()
font = cv2.FONT_HERSHEY_SIMPLEX
print(" * Loading Keras model...")
get_model()
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        faces = facec.detectMultiScale(fr, 1.3, 5)
        for (x, y, w, h) in faces:
            fc = fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (119, 119))  #(119, 119, 3)
            print(roi.shape)
            roi = np.expand_dims(roi, axis=0)
            print(roi.shape)   #(1,119,119,3)
            pred = model.predict(roi)
            #pred = model.predict(roi[np.newaxis, :, :, np.newaxis])
            print(pred)  #[[0.00000e+00 0.00000e+00 2.03737e-20 1.00000e+00 0.00000e+00 0.00000e+000.00000e+00 0.00000e+00]]
            a = pred[0].tolist()
            pred = str(int(a[0] * 1.3 + a[1] * 4.2 + a[2] * 10.1 + a[3] * 19.8 + a[4] * 27.8 + a[5] * 38.1 + a[6] * 51.6 + a[7] * 71.8))
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(fr, pred, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
