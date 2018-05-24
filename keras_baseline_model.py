import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import Sequential
from keras.layers import Dense, Input, Flatten, Activation, Dropout
import glob
import os
import numpy as np
from tqdm import tqdm
from google_landmarks_dataset import GoogleLandmarkTestGenerator
import google_landmarks_dataset

# load data
data_path = './data/landmarks_recognition/'
image_path = './data/landmarks_recognition/128_128/'
image_train_path = os.path.join(image_path, 'train')
image_test_path = os.path.join(image_path, 'test')

num_classes = 14951
batch_size = 62

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        image_train_path,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical')

label_map = train_generator.class_indices
inv_label_map = {v: k for k, v in label_map.items()}
print(inv_label_map)

model = Sequential()
model.add(Flatten(input_shape=(128, 128, 3)))  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())
model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // 32,
        epochs=100)


test_generator = GoogleLandmarkTestGenerator(data_path, size=(128, 128), batch_size=batch_size)

label_predictions = []
prob_predictions = []
idxs = []

count = 0
with open('./output/keras_baseline_submission.csv', 'w') as f:
  f.write('id,landmarks\n')
  for idx_batch, image_batch in test_generator:
    # count +=1
    # if count > 5:
    #   break
    predictions = model.predict(image_batch, batch_size=batch_size, verbose=1)  
    predict_label = np.argmax(predictions, axis=1)  
    batch_label_predictions = list(map(lambda x: inv_label_map[x],predict_label))
    batch_prob_predictions = np.max(predictions, axis=1)

    for i in range(len(idx_batch)):
      f.write('{},{} {}\n'.format(idx_batch[i], batch_label_predictions[i], batch_prob_predictions[i]))
    label_predictions.extend(batch_label_predictions)
    prob_predictions.extend(batch_prob_predictions)
    idxs.extend(idx_batch)

  # write missing image
  missing_data = google_landmarks_dataset.load_missing_test_data(data_path, size=(128, 128))
  for _, row in missing_data.iterrows():
    f.write('{}\n'.format(row['id']))