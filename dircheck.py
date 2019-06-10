import os
import imageio
from scipy import misc
from os import listdir
from random import shuffle
import cv2
import numpy as np
import math

#trainimages = os.listdir("C:/Users/thali/Desktop/data1232s/train")
#print(len(trainimages))
testimages = os.listdir("C:/Users/thali/Desktop/mix2s/train")
print(len(testimages))
age = 1 + math.floor(np.random.randn() * 2 + 0.5)
print(age)
print(str(np.clip(age, 0, 100)))
