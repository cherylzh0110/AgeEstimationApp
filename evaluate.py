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
from model import get_model

model = get_model(model_name = "ResNet")
model.load_weights('weights.18-8.49.h5')
# read model and calculate MAE value of testing dataset
input_path = "C:/Users/thali/Pycharm Projects/CNNApp/models"

def get_model():
    global model
    model = load_model('weights.18-8.49.h5')
    model._make_predict_function()
#get_model()
x_test = []
Y_age =[]
img_height,img_width = 119,119
target_size=(img_width,img_height)

base_path = "C:/Users/thali/Desktop/data2s/"
train_path = "C:/Users/thali/Desktop/data2s/train"
test_path = "C:/Users/Cheryl Zhang/Desktop/data2s/test"
os.chdir(test_path)
testimages = os.listdir()
#testimages = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]

for file in testimages:
    Y_age.append(int(file.split("_")[0]))
print(Y_age)

for pic in testimages:
    face =misc.imread(pic)
    face =cv2.resize(face, target_size)
    #print(face.shape)   (119,119,3)
    x_test.append(face)

x_test = np.squeeze(x_test)  #(1,119,119,3)
x_test = x_test.astype('float32')
x_test /= 255
print(x_test)

y_pred1 = model.predict(x_test)   #multi-dim
#print(y_pred1)  #multi-dim
ages = np.arange(0, 101).reshape(101, 1)
predicted_ages = y_pred1.dot(ages).flatten()  # list of predicted ages
print(predicted_ages)

def calAgeList(y_pred1): # Return a list of predictions, given multi-dim input
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = y_pred1.dot(ages).flatten()
    return predicted_ages

# average age prediction which a list of ages
def ensembleAge(result1,result2,result3):
    agelist = list(np.array(result1) + np.array(result2) + np.array(result3))
    agelist = [number/3 for number in agelist]
    return agelist

#print("MAE value is : " + str(maeCal(y_pred1, Y_age)))
def maeCal(Y_age,predicted_ages):
    #print(len(predicted_ages))  # Y_age: a list of actual age, y_pred: a list of predicted ages
    return np.mean(np.abs(np.array(predicted_ages) - np.array(Y_age)))

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


history = pd.read_hdf(input_path, "history")
plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.savefig(os.path.join(input_path, "loss.png"))
plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history['val_age_mae'], label='Training Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()
plt.savefig(os.path.join(input_path, "mae.png"))


if __name__ == '__main__':
    # compute confusion matrix
    cnf_matrix = confusion_matrix(Y_age,predicted_ages)
    np.set_printoptions(precision=2)
    # plot confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classestitle='Confusion matrix')
    plt.show()
    
 
#print("MAE value is : " + str(maeCal(y_pred1, Y_age)))
def maeCal(Y_age,predicted_ages):
    #print(len(predicted_ages))  # Y_age: a list of actual age, y_pred: a list of predicted ages
    return np.mean(np.abs(np.array(predicted_ages) - np.array(Y_age)))

