import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from scipy import misc
from os import listdir
from random import shuffle
import cv2
from keras.models import Sequential, Model
from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.metrics import confusion_matrix
import itertools
import model as get_model
from preprocess import transferAge8
from plothistory import plot_acc, plot_mae
from evaluate import plot_confusion_matrix

# dataset size, image pixels, batches, epoches(each of them has 3 options)
#X_train_orig is the 4-dim list, X_train shape: (23708, 32, 32, 3)
X_train_orig = []
# change directory
os.chdir('C:/Users/thali/Desktop/UTKFace')
onlyfiles = os.listdir()
shuffle(onlyfiles)
for pic in onlyfiles:
    face =misc.imread(pic)
    face =cv2.resize(face, (224, 224))
    X_train_orig.append(face)

numclasses = 8
Y_age =[]
for file in onlyfiles:
    Y_age.append(file.split("_")[0])
Y_train_orig = transferAge8(Y_age)

categorical_labels = np.eye(numclasses)[Y_train_orig]
class_labels = np.argmax(categorical_labels, axis=1)
(x_train, y_train), (x_test, y_test) = (X_train_orig[:20000],categorical_labels[:20000]) , (X_train_orig[20000:22000] , categorical_labels[20000:22000])
x_train = np.squeeze(x_train)
x_train = x_train.astype('float32')
x_train /= 255
x_test = np.squeeze(x_test)
x_test = x_test.astype('float32')
x_test /= 255

model = get_model(model_name="VGG16")
model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss="categorical_crossentropy", metrics=['accuracy'])
checkpoint = [ModelCheckpoint("model_weightsvgg.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')]

history = model.fit(x_train, y_train, validation_split=0.1,epochs = 25, batch_size = 32, callbacks=checkpoint, verbose=1)
# list all data in history
# print(history.history.keys())
# summarize history for accuracy
plot_acc(history)

y_pred = model.predict(x_test)
y_pred = [np.argmax(probas) for probas in y_pred]
y_test = [np.argmax(probas) for probas in y_test]
class_names = [0,1,2,3,4,5,6,7]
# compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()

