import os, os.path, shutil
from random import shuffle
import numpy as np
import math
#create file consisting of randomly splitting training and testing data(0.9 training dataset)
'''
new_path = "C:/Users/thali/Desktop/crop"
folder_path = 'C:/Users/thali/Desktop/crop0'
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
print(len(files))
for f in files:
    if np.random.rand(1) < 0.1:
        shutil.move(folder_path + '/'+ f, new_path + '/test/'+ f)
    else:
        shutil.move(folder_path + '/'+ f, new_path + '/train/'+ f)
'''

def transferAge12(Y_age):
    Y_label = []
    for i in Y_age:
        i = int(i)
        if i <= 2:
            Y_label.append(0)
        elif (i>2) and (i<=6):
            Y_label.append(1)
        elif (i>6) and (i<15):
            Y_label.append(2)
        elif (i>=15) and (i<22):
            Y_label.append(3)
        elif (i>=22) and (i<27):
            Y_label.append(4)
        elif (i>=27) and (i<32):
            Y_label.append(5)
        elif (i>=32) and (i<40):
            Y_label.append(6)
        elif (i>=40) and (i<48):
            Y_label.append(7)
        elif (i>=48) and (i<55):
            Y_label.append(8)
        elif (i>=55) and (i<70):
            Y_label.append(9)
        elif i>=70:
            Y_label.append(10)
    return Y_label

def transferAge8(Y_age):
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
'''
# Create files grouped by ages(101 groups in total)
train_path = "C:/Users/thali/Desktop/crop/train"
test_path = "C:/Users/thali/Desktop/crop/test"
trainimages = [f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))]
testimages = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
print(len(trainimages))
print(len(testimages))

shuffle(trainimages)
for image in trainimages:
    folder_name = image.split("_")[0]
    #perform label augmentation via generating samples from the normal distribution
    #add Gaussian noise to labels (ages)
    #age = int(folder_name) + math.floor(np.random.randn() * 2 + 0.5)
    #folder_name = np.clip(age, 0, 100)
    new_path = os.path.join(train_path, folder_name)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    old_image_path = os.path.join(train_path, image)
    new_image_path = os.path.join(new_path, image)
    shutil.move(old_image_path, new_image_path)

shuffle(testimages)
for image in testimages:
    folder_name = image.split("_")[0]
    new_path = os.path.join(test_path, folder_name)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    old_image_path = os.path.join(test_path, image)
    new_image_path = os.path.join(new_path, image)
    shutil.move(old_image_path, new_image_path)
'''
