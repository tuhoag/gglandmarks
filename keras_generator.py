from google_landmarks_dataset import GoogleLandmarkTestGenerator, create_train_and_test_generators, GoogleLandmarkGenerator
import os

# load data
image_path = './data/landmarks_recognition/'
image_train_path = os.path.join(image_path, 'train')
image_test_path = os.path.join(image_path, 'test')
image_archive_test_path = os.path.join(image_path, 'test1')
num_classes = 14951
batch_size = 62

# create_train_and_test_generators(image_path, (128,128))
generator = GoogleLandmarkGenerator(image_path, (128, 128))