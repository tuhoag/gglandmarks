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
from google_landmarks_dataset import GoogleLandmarkTestGenerator
from app.models import MyDenseNet
from app.datasets import GoogleLandmarkGenerator

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
image_width = 48
image_height = 48
image_channel = 3
image_shape=(image_width, image_height, image_channel)
model_weights_file = './weights/densenet121.h5'

model = MyDenseNet(image_shape, num_classes)
print(model.summary())
generator = GoogleLandmarkGenerator(data_path, (128, 128))
model.train_and_validate(generator=generator, epochs=1, validation_split=0.1, batch_size=batch_size)
# test_generator = GoogleLandmarkTestGenerator(data_path, 
#   size=(image_original_width, image_original_height), 
#   target_size=(image_width, image_height), 
#   batch_size=batch_size)

# label_predictions = []
# prob_predictions = []
# idxs = []

# count = 0
# with open(output_path, 'w') as f:
#   f.write('id,landmarks\n')
#   for idx_batch, image_batch in test_generator:
#     # count +=1
#     # if count > 5:
#     #   break
#     predictions = model.predict(image_batch, batch_size=batch_size, verbose=1)  
#     predict_label = np.argmax(predictions, axis=1)  
#     batch_label_predictions = list(map(lambda x: inv_label_map[x],predict_label))
#     batch_prob_predictions = np.max(predictions, axis=1)

#     for i in range(len(idx_batch)):
#       f.write('{},{} {:.2f}\n'.format(idx_batch[i], batch_label_predictions[i], batch_prob_predictions[i]))
#     label_predictions.extend(batch_label_predictions)
#     prob_predictions.extend(batch_prob_predictions)
#     idxs.extend(idx_batch)

#   # write missing image
#   missing_data = google_landmarks_dataset.load_missing_test_data(data_path, size=(128, 128))
#   for _, row in missing_data.iterrows():
#     f.write('{},\n'.format(row['id']))