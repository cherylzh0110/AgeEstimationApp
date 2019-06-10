from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import os
import matplotlib.pyplot as plt
from model import get_model
from plothistory import plot_acc, plot_mae
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser
from random import shuffle
import cv2
from scipy import misc

batch_size = 32
img_height,img_width = 119,119
target_size=(img_width,img_height)
base_path = "C:/Users/thali/PycharmProjects/CNNApp"
train_path = "C:/Users/thali/Desktop/mix2s/train"
test_path = "C:/Users/thali/Desktop/mix2s/test"
output_path = "C:/Users/thali/PycharmProjects/CNNApp/models"

X_train_orig = []
# change directory
os.chdir(train_path)
onlyfiles = os.listdir()
shuffle(onlyfiles)
for pic in onlyfiles:
    face =misc.imread(pic)
    face =cv2.resize(face, (119, 119))
    X_train_orig.append(face)

Y_age =[]
for file in onlyfiles:
    Y_age.append(int(file.split("_")[0]))
categorical_labels = np.eye(101)[Y_age]
(x_train, y_train), (x_test, y_test) = (X_train_orig,categorical_labels) , (X_train_orig, categorical_labels)
x_train = np.squeeze(x_train)
x_train = x_train.astype('float32')
x_train /= 255
x_test = np.squeeze(x_test)
x_test = x_test.astype('float32')
x_test /= 255

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        preprocessing_function=get_random_eraser(v_l=0, v_h=255))
# all images are randomly erased before standard augmentation done by ImageDataGenerator.
training_generator = MixupGenerator(x_train, y_train, batch_size=batch_size, alpha=0.2,
                                            datagen=train_datagen)()

num_classes = 101
model = get_model("ResNet")

def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 101, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 101, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae

callbacks = [ModelCheckpoint("weightsmix.{epoch:02d}-{val_age_mae:.2f}.h5",
                                 monitor="val_age_mae",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="min")]

#sgd = SGD(lr=1e-4, momentum=0.9, nesterov=False)
adam = Adam(lr=0.001)    #adam is better than sgd
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=[age_mae])
# fine-tune the model

history = model.fit_generator(generator=training_generator,
                                   steps_per_epoch=20500 // batch_size,
                                   validation_data=(x_test, y_test),
                                   epochs=40, verbose=1,
                                   callbacks=callbacks)

np.savez(output_path + "/history.npz", history=history.history)

os.chdir(output_path)
plot_mae(history)
'''
def maeCal(y_test,predicted_ages):
    #print(len(predicted_ages))  # Y_age: a list of actual age, y_pred: a list of predicted ages
    return np.mean(np.abs(np.array(predicted_ages) - np.array(int(x) for x in y_test)))
print(maeCal(y_test, predicted_ages))
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
'''

