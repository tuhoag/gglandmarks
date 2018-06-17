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
from gglandmarks.datasets.google_landmarks_dataset import GoogleLandmarkTestGenerator
from gglandmarks.models import MyDenseNet, MyBasicModel
from gglandmarks.datasets import GoogleLandmarkDataset

# load data
data_path = './data/landmarks_recognition/'
image_path = './data/landmarks_recognition/128_128/'
image_train_path = os.path.join(image_path, 'train')
image_test_path = os.path.join(image_path, 'test')
output_file_name = 'keras_vgg_16.csv'
output_folder = './output/'
output_path = os.path.join(output_folder, output_file_name)

num_classes = 14951
batch_size = 64
image_original_width = 128
image_original_height = 128
image_width = 128
image_height = 128
image_channel = 3
batch_shape=(None, image_width, image_height, image_channel)


dataset = GoogleLandmarkDataset(data_path, (image_original_width, image_original_height), images_count_min=30000)
print(dataset.train_df.shape)
print(dataset.num_classes)

model = MyBasicModel(batch_shape, dataset.num_classes, logdir='./logs/')
model.fit3(dataset)