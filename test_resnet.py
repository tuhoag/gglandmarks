from gglandmarks.datasets.google_landmarks_dataset import GoogleLandmarkTestGenerator
from gglandmarks.models import MyResNets
from gglandmarks.datasets import GoogleLandmarkDataset
import os

# load data
data_path = './data/landmarks_recognition/'
image_path = './data/landmarks_recognition/128_128/'
image_train_path = os.path.join(image_path, 'train')
image_test_path = os.path.join(image_path, 'test')
output_file_name = 'keras_vgg_16.csv'
output_folder = './output/'
output_path = os.path.join(output_folder, output_file_name)

num_classes = 14951
batch_size = 32
image_original_width = 128
image_original_height = 128
image_width = 48
image_height = 48
image_channel = 3
batch_shape = (None, image_width, image_height, image_channel)

MyResNets.finetune(data_path, (image_original_width, image_original_height))
