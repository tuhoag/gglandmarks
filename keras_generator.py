from app.datasets import GoogleLandmarkGenerator
import matplotlib.pyplot as plt
import pandas as pd
import os

# load data
image_path = './data/landmarks_recognition/'
image_train_path = os.path.join(image_path, 'train')
image_test_path = os.path.join(image_path, 'test')
image_archive_test_path = os.path.join(image_path, 'test1')
num_classes = 14951
batch_size = 64
epochs = 10
target_size = (64, 64)
# create_train_and_test_generators(image_path, (128,128))
generator = GoogleLandmarkGenerator(image_path, (128, 128), images_count_min=1000)
stats = generator.get_frequent_landmarks()

print(stats)
stats.to_csv('stats.csv')

gen = generator.get_train_validation_generator(batch_size, target_size, 0.1)
for i in range(epochs):
  train_gen, test_gen = next(gen)

  # print(train_gen, ' ', test_gen)
  
  # print(next(train_gen))
  # print(next(test_gen))
  countx = 0
  county = 0
  j = 0  
  for x, y in test_gen:
    
    countx += x.shape[0]
    county += y.shape[0]
    print('{}/{} - x: {}, y: {}'.format(j, len(test_gen), x.shape, y.shape))  
    j+=1
    
  print('total x: {}, total y: {}'.format(countx, county))  