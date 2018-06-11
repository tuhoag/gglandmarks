import math
import numpy as np
from gglandmarks.utils import load_image
from keras.utils import to_categorical

class DataGenerator(object):
    def __init__(self, paths, landmarks, encoder, batch_size, target_size):
        self.batch_size = batch_size
        self.target_size = target_size
        self.paths = paths
        self.landmarks = landmarks
        self.encoder = encoder
        self.num_batches = math.ceil(len(paths) / batch_size)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        print('{}/{}'.format(self.current_idx + 1, self.num_batches))
        if self.current_idx >= self.num_batches:
            raise StopIteration()

        start_idx = self.current_idx * self.batch_size
        end_idx = (self.current_idx + 1) * self.batch_size
        batch_paths = self.paths[start_idx:end_idx]
        batch_landmarks = self.landmarks[start_idx:end_idx]

        X_temp = []
        for path in batch_paths:
            image_array = load_image(path, self.target_size)
            if image_array is not None:
                image_array = image_array.reshape(
                    self.target_size[0], self.target_size[1], 3)
                X_temp.append(image_array)
        # print(self.encoder.classes_)
        # print(batch_landmarks)
        Y = self.encoder.encode(batch_landmarks)
        
        # print('transform: {}'.format(Y))        
        self.current_idx += 1

        return np.asarray(X_temp), Y

    def __len__(self):
        return self.num_batches
