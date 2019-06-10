import cv2
from keras.models import Sequential, load_model
import numpy as np

facec = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

def get_model():
    global model
    model = load_model('./models/utk.h5')
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
        success, fr = self.video.read()
        faces = facec.detectMultiScale(fr, 1.3, 5)
        for (x, y, w, h) in faces:
            fc = fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (64, 64))
            roi = np.expand_dims(roi, axis=0)
            #print(roi)
            prediction1 = model.predict(roi) #[[0.00000e+00 0.00000e+00 2.03737e-20 1.00000e+00 0.00000e+00 0.00000e+000.00000e+00 0.00000e+00]]
            #print(prediction1)
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = prediction1[1].dot(ages).flatten()
            #print(predicted_ages)
            pred = str(int(predicted_ages[0]))
            #pred = str(a[0] * 1 + a[1] * 4.5 + a[2] * 10.5 + a[3] * 19 + a[4] * 29 + a[5] * 40 + a[6] * 53 + a[7] * 70)
            print(pred)
            #pred = model.predict(roi[np.newaxis, :, :, np.newaxis])
            #print(pred)  #[[0.00000e+00 0.00000e+00 2.03737e-20 1.00000e+00 0.00000e+00 0.00000e+000.00000e+00 0.00000e+00]]
            #pred = str(a[0] * 1 + a[1] * 4.5 + a[2] * 10.5 + a[3] * 19 + a[4] * 29 + a[5] * 40 + a[6] * 53 + a[7] * 70)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(fr, str(pred), (250, 450), cv2.FONT_HERSHEY_COMPLEX,1, (0,0,255))
        ret, jpeg = cv2.imencode('.jpg', fr)
        #OpenCV captures rare image, so images need to be encoded into jpg to correctly display the video stream
        return jpeg.tobytes()
