import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np


output_path = "C:/Users/thali/PycharmProjects/CNNApp/models"
def main():
    df = np.load(output_path, "history")
    plt.figure(figsize=(20,10))
    plt.subplot(1, 2, 1)
    plt.suptitle('Optimizer : Adam', fontsize=10)
    plt.ylabel('Loss', fontsize=16)
    plt.plot(df['loss'], label='Training Loss')
    plt.plot(df['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(output_path, "loss.png"))
    plt.subplot(1, 2, 2)
    plt.ylabel('MAE', fontsize=16)
    plt.plot(df['val_age_mae'], label='Validation MAE')
    plt.plot(df['val_acc'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(os.path.join(output_path, "mae.png"))

def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(output_path + "/acc.png")
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(output_path + "/loss.png")
    print(history.history['val_acc'])
    print(history.history['val_loss'])

def plot_mae(history):
    plt.ylabel('Loss', fontsize=16)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(output_path + "/loss.png")

    plt.plot(history.history['age_mae'], label='Training MAR')
    plt.plot(history.history['val_age_mae'], label='Validation MAE')
    plt.title('model MAE')
    plt.ylabel('MAE', fontsize=16)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(output_path + "/MAE.png")
    print(history.history['val_age_mae'])
    print(history.history['val_loss'])

if __name__ == '__main__':
    main()
