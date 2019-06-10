import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import os
import imageio
from scipy import misc
from os import listdir
from random import shuffle
import cv2
import keras
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.vgg16 import VGG16
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Dropout
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

# dataset size, image pixels, batches, epoches(each of them has 3 options)
#X_train_orig is the 4-dim list, X_train shape: (23708, 32, 32, 3)
X_train_orig = []
# change directory
os.chdir('C:/Users/thali/Desktop/Project 90055/utkface-new/UTKFace')
onlyfiles = os.listdir()
shuffle(onlyfiles)
for pic in onlyfiles:
    face =misc.imread(pic)
    face =cv2.resize(face, (119, 119))
    X_train_orig.append(face)

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

Y_age =[]
for file in onlyfiles:
    Y_age.append(file.split("_")[0])
Y_train_orig = transferAge(Y_age)
categorical_labels = np.eye(8)[Y_train_orig]  #[0,0,1,0,0,0,0,0] for label 2
class_labels = np.argmax(categorical_labels, axis=1)

(x_train, y_train), (x_test, y_test) = (X_train_orig[:20000],categorical_labels[:20000]) , (X_train_orig[20000:] , categorical_labels[20000:])
(x_valid , y_valid) = (x_test[0:1800], y_test[0:1800])

x_train = np.squeeze(x_train)
x_train = x_train.astype('float32')
x_train /= 255
x_valid = np.squeeze(x_valid)
x_valid = x_valid.astype('float32')
x_valid /= 255

vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(119, 119, 3), classes = 8)
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(8, activation='softmax'))

model = Sequential()
model.add(vgg16_model)
model.add(top_model)
checkpoint = [ModelCheckpoint("model_weights.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')]
model.compile(optimizer=SGD(lr=1e-3, momentum=0.9), loss="categorical_crossentropy", metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.1,epochs = 25, batch_size = 32, callbacks=checkpoint)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

'''
from keras.callbacks import Callback
class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_batch_end(self, batch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        plt.figure()
        plt.plot(xc, loss)
        plt.plot(xc, val_loss)
        plt.show()

model.fit(x_train,
         y_train,
         epochs=20, batch_size=32,
         validation_data=(x_valid, y_valid),callbacks=[TestCallback((x_valid, y_valid))])

from keras_tqdm import TQDMNotebookCallback
myepochs = 1
mybatch = 32
seqModel =model.fit(x_train, y_train,
          batch_size      = mybatch,
          epochs          = myepochs,
          validation_data = (x_valid, y_valid),
          shuffle         = True,
          verbose=0, callbacks=[TQDMNotebookCallback()]) #for visualization
seqModel
# visualizing losses and accuracy
train_loss = seqModel.history['loss']
val_loss   = seqModel.history['val_loss']
train_acc  = seqModel.history['acc']
val_acc    = seqModel.history['val_acc']
xc         = range(myepochs)

plt.figure()
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.show()
#plt.show(block=True)
#plt.interactive(False)

# list all data in history
history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''

os.chdir('C:/Users/thali/PycharmProjects/CNNApp')
target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./models/vgg16.h5')
model.save_weights('./models/vgg16.h5')

