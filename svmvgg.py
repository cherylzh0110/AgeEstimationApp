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
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D,BatchNormalization,GlobalMaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.applications.vgg16 import VGG16
from keras.regularizers import l2
from keras import applications
from keras.optimizers import SGD, Adam
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
categorical_labels = np.eye(8)[Y_train_orig]
class_labels = np.argmax(categorical_labels, axis=1)

(x_train, y_train), (x_test, y_test) = (X_train_orig[:15000],categorical_labels[:15000]) , (X_train_orig[15000:] , categorical_labels[15000:])
(x_valid , y_valid) = (x_test[2000:3500], y_test[2000:3500])

x_train = np.squeeze(x_train)
x_train = x_train.astype('float32')
x_train /= 255
x_valid = np.squeeze(x_valid)
x_valid = x_valid.astype('float32')
x_valid /= 255

'''
from keras.layers import Input
input_img = Input(shape=(img_height*img_width*3,))
encoding_dim = 32
# this is the bottleneck vector
encoded = Dense(encoding_dim, activation='relu')(input_img)
# this is the decoded layer, with the same shape as the input
decoded = Dense(img_height*img_width*3, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
'''

vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(119, 119, 3), classes = 8)
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
#top_model.add(Dense(8, activation='softmax'))
top_model.add(Dense(8), kernel_regularizer=l2(0.01))
top_model.add(Activation('linear'))

model = Sequential()
model.add(vgg16_model)
model.add(top_model)
model.compile(loss='hinge', optimizer='adadelta', metrics=['accuracy'])
#model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(x_train, y_train, validation_split=0.1,epochs = 30, batch_size = 64, verbose=0)

'''
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


os.chdir('C:/Users/thali/PycharmProjects/CNNApp')
target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./models/vae.h5')
model.save_weights('./models/vae.h5')
'''
