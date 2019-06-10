import random
import math
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from keras.utils import Sequence, to_categorical
import Augmentor


def get_transform_func():
    p = Augmentor.Pipeline()
    p.flip_left_right(probability=0.5)
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.zoom_random(probability=0.5, percentage_area=0.95)
    p.random_distortion(probability=0.5, grid_width=2, grid_height=2, magnitude=8)
    p.random_color(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_contrast(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_brightness(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_erasing(probability=0.5, rectangle_area=0.2)

    def transform_image(image):
        image = [Image.fromarray(image)]
        for operation in p.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                image = operation.perform_operation(image)
        return image[0]
    return transform_image


class FaceGenerator(Sequence):
    def __init__(self, utk_dir):
        self.images = []
        image_dir = Path(utk_dir)
        for image_path in image_dir.glob("*.jpg"):
            image_name = image_path.name
            age = min(100, int(image_name.split("_")[0]))
            if image_path.is_file():
                self.images.append([str(image_path), age])
        self.image_num = len(self.images)
        self.indices = np.random.permutation(self.image_num)
        self.transform_image = get_transform_func()

    def __len__(self):
        return self.image_num // 32

    def __getitem__(self, idx):
        batch_size = 32
        image_size = 119
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, 1), dtype=np.int32)
        sample_indices = self.indices[idx * batch_size:(idx + 1) * batch_size]
        for i, sample_id in enumerate(sample_indices):
            image_path, age = self.images[sample_id]
            image = cv2.imread(str(image_path))
            x[i] = self.transform_image(cv2.resize(image, (image_size, image_size)))
            age += math.floor(np.random.randn() * 2 + 0.5)   #Simply add Gaussian noise to labels
            y[i] = np.clip(age, 0, 100)
        return x, to_categorical(y, 101)

    def on_epoch_end(self):
        self.indices = np.random.permutation(self.image_num)

class ValGenerator(Sequence):
    def __init__(self, utk_dir):
        self.images = []
        image_dir = Path(utk_dir)
        for image_path in image_dir.glob("*.jpg"):
            image_name = image_path.name
            age = min(100, int(image_name.split("_")[0]))
            if image_path.is_file():
                self.images.append([str(image_path), age])
        self.image_num = len(self.images)

    def __len__(self):
        return self.image_num // 32

    def __getitem__(self, idx):
        batch_size = 32
        image_size = 119
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, 1), dtype=np.int32)
        for i in range(batch_size):
            image_path, age = self.images[idx * batch_size + i]
            image = cv2.imread(str(image_path))
            x[i] = cv2.resize(image, (image_size, image_size))
            y[i] = age
        return x, to_categorical(y, 101)



