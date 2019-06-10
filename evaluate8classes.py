from keras.models import Sequential, load_model
from keras import backend as K
import os
import imageio
from scipy import misc
from os import listdir
from random import shuffle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.backend import clear_session
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#model = get_model(model_name = "ResNet")
#model.load_weights('weights.04-21.50.h5')
# read model and calculate MAE value of testing dataset
#class_names = list(range(1,8))
class_names = ["0-1","2-6","7-14","15-23","24-34","35-44","45-59","60-101",]
#resnet: 4.28
#vgg16: 4.78
#ensemble of vgg16 and resnet: 3.95
def get_model():
    global model
    model = load_model('./models/resnet3.h5')
    model._make_predict_function()
get_model()

x_test = []
Y_age =[]
img_height,img_width = 119,119
target_size=(img_width,img_height)
input_path = "C:/Users/Cheryl Zhang/PycharmProjects/WebApp"
base_path = "C:/Users/thali/Desktop/data2s/"
test_path = "C:/Users/Cheryl Zhang/Desktop/data2s/test"

os.chdir(test_path)
testimages = os.listdir()
#testimages = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
for file in testimages:
    Y_age.append(int(file.split("_")[0]))
#print(Y_age)

for pic in testimages:
    face =misc.imread(pic)
    face =cv2.resize(face, target_size)
    #print(face.shape)   (119,119,3)
    x_test.append(face)

x_test = np.squeeze(x_test)  #(1,119,119,3)
x_test = x_test.astype('float32')
x_test /= 255
y_pred = model.predict(x_test)
alist = [1.3,4.2,10.1,19.8,27.8,38.1,51.6,71.8]
ages = np.reshape(alist,(8, 1))
predicted_ages = y_pred.dot(ages).flatten()  # list of predicted ages
#print(predicted_ages)
clear_session()
#predicted_ages0s = [int(y) for y in predicted_ages]


os.chdir(input_path)
def get_model():
    global model
    model = load_model('./models/model64.h5')
    model._make_predict_function()
get_model()

y_pred1 = model.predict(x_test)
alist = [1.3,4.2,10.1,19.8,27.8,38.1,51.6,71.8]
ages = np.reshape(alist,(8, 1))
predicted_ages1 = y_pred1.dot(ages).flatten()  # list of predicted ages
#print(predicted_ages1)
#predicted_ages1s = [int(y) for y in predicted_ages1]

# ensemble of vgg16 and resnet prediction
def ensembleAge(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)
predicted_ages2 = [ensembleAge(i) for i in zip(predicted_ages,predicted_ages1)]
print(predicted_ages2)  # a list of predicted ages based on ensemble model(float value)

def maeCal(Y_age,result_list):
    #print(len(predicted_ages))  # Y_age: a list of actual age, y_pred: a list of predicted ages
    return np.mean(np.abs(np.array(result_list) - np.array(Y_age)))
#print("MAE value of ensemble networks is : " + str(maeCal(Y_age,predicted_ages2)))

predicted_ages2s = [int(y) for y in predicted_ages2]
#print(predicted_ages2s)  #is int value list

#save classification report to csv file
report = classification_report(Y_age, predicted_ages2s)
#print(accuracy_score(Y_age, predicted_ages2s))   #0.1323
print(report)
print(confusion_matrix(Y_age, predicted_ages2s))

def calAgeList(y_pred1): # Return a list of predictions, given multi-dim input
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = y_pred1.dot(ages).flatten()
    return predicted_ages

# average age prediction which a list of ages
def ensembleAge(result1,result2,result3):
    agelist = list(np.array(result1) + np.array(result2) + np.array(result3))
    agelist = [number/3 for number in agelist]
    return agelist

# show the confusion matrix of our predictions
from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
y_pred = [int(np.argmax(prob)) for prob in y_pred]
def transferAge(Y_age):
    Y_label = []
    for i in Y_age:
        i = int(i)
        if i <= 2:
            Y_label.append(0)
        elif (i>2) and (i<=6):
            Y_label.append(1)
        elif (i>6) and (i<15):
            Y_label.append(2)
        elif (i>=15) and (i<24):
            Y_label.append(3)
        elif (i>=24) and (i<35):
            Y_label.append(4)
        elif (i>=35) and (i<45):
            Y_label.append(5)
        elif (i>=45) and (i<60):
            Y_label.append(6)
        elif i>=60:
            Y_label.append(7)
    return Y_label
y_age = transferAge(Y_age)
# compute confusion matrix
cnf_matrix = confusion_matrix(y_age,y_pred)
np.set_printoptions(precision=2)

# plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()