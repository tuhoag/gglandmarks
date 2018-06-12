import keras
from keras.applications import VGG16, DenseNet121
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import Sequential
from keras.layers import Dense, Input, Flatten, Activation, Dropout
from keras.models import Model, load_model
import glob
import os
import numpy as np
from tqdm import tqdm
from .abstract_model import AbstractModel

class MyBasicModel(object):
    def __init__(self, input_shape, num_classes, weight_cache_folder=os.getcwd()):
        super().__init__(input_shape, num_classes, weight_cache_folder)

    def create_model(self, input_shape, num_classes):
 
       