from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Model
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import os
import matplotlib.pyplot as plt
from model import get_model
from plothistory import plot_acc, plot_mae
from random_eraser import get_random_eraser
from generator import FaceGenerator, ValGenerator
import numpy as np
from keras.optimizers import SGD, Adam

batch_size = 32
img_height,img_width = 119,119
target_size=(img_width,img_height)
base_path = "C:/Users/thali/Desktop/data/"
train_path = "C:/Users/thali/Desktop/data1232s/train"
test_path = "C:/Users/thali/Desktop/data1232s/test"
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

train_generator = FaceGenerator(utk_dir=train_path)
# only rescaling
validation_generator = ValGenerator(utk_dir=test_path)

num_classes = 101
model = get_model("ResNet")

def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 101, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 101, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae
callbacks = [LearningRateScheduler(schedule=Schedule(60, initial_lr=0.001)),
                 ModelCheckpoint("weightslab.{epoch:03d}-{val_loss:.3f}-{val_age_mae:.3f}.h5",
                                 monitor="val_age_mae",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="min")
                 ]

#sgd = SGD(lr=1e-4, momentum=0.9, nesterov=False)
adam = Adam(lr=0.001)    #adam is better than sgd
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=[age_mae])
# fine-tune the model
history = model.fit_generator(generator=train_generator,
                                epochs=65,
                                validation_data = validation_generator,
                                verbose=1,
                                callbacks=callbacks
                                )
np.savez(output_path + "/history2.npz", history=history.history)

os.chdir(output_path)
plot_mae(history)

