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
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D,BatchNormalization,GlobalMaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.applications.resnet50 import ResNet50
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras import backend as K
from preprocess import transferAge8
# dataset size, image pixels, batches, epoches(each of them has 3 options)
#X_train_orig is the 4-dim list, X_train shape: (23708, 119, 119, 3)
X_train_orig = []
Y_age =[]
# change directory
bath_path = "C:/Users/thali/PycharmProjects/CNNApp"
output_path = "C:/Users/thali/PycharmProjects/CNNApp/models"
train_path = "C:/Users/thali/Desktop/data2s/train"
test_path = "C:/Users/thali/Desktop/data2s/test"
output_path = "C:/Users/thali/PycharmProjects/CNNApp/models"

class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008
os.chdir(train_path)
onlyfiles = os.listdir()
shuffle(onlyfiles)
for pic in onlyfiles:
    face =misc.imread(pic)
    face =cv2.resize(face, (119, 119))
    X_train_orig.append(face)
for file in onlyfiles:
    Y_age.append(file.split("_")[0])
Y_train_orig = transferAge8(Y_age)
categorical_labels = np.eye(8)[Y_train_orig]
class_labels = np.argmax(categorical_labels, axis=1)

(x_train, y_train), (x_test, y_test) = (X_train_orig[:20000],categorical_labels[:20000]) , (X_train_orig[20000:] , categorical_labels[20000:])
x_train = np.squeeze(x_train)
x_train = x_train.astype('float32')
x_train /= 255
os.chdir(bath_path)
from keras import applications
from keras.applications.resnet50 import ResNet50
img_height,img_width = 119,119
num_classes = 8
class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008
base_model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape= (img_height,img_width,3))
x = base_model.output
x = GlobalMaxPooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
#dense = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)
os.chdir(output_path)

callbacks = [LearningRateScheduler(schedule=Schedule(30, initial_lr=0.001)),
                 ModelCheckpoint("weights8.{epoch:02d}-{val_acc:.2f}.h5",
                                 monitor="val_acc",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="max")
                 ]
from keras.optimizers import SGD, Adam
#sgd = SGD(lr=1e-3, momentum=0.9, nesterov=False)
adam = Adam(lr=0.001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(x_train, y_train, validation_split=0.2, epochs = 5, batch_size = 32, verbose=1)
history = model.fit(x_train, y_train, validation_split=0.1,epochs = 30, batch_size = 32, verbose=1, callbacks=callbacks)
# list all data in history
np.savez(output_path + "/historyRes.npz", history=history.history)
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

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from plothistory import plot_acc
from evaluate import plot_confusion_matrix

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
report = classification_report(y_test, y_pred)
print(accuracy_score(y_test, y_pred))   #0.7
print(report)
