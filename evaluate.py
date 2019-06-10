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


f = np.load('./models/history.npz')
dic = f['history']
plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(dic['loss'], label='Training Loss')
plt.plot(dic['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.savefig(os.path.join(input_path, "loss.png"))
plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(dic['val_age_mae'], label='Training Accuracy')
plt.plot(dic['val_acc'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()
plt.savefig(os.path.join(input_path, "mae.png"))


