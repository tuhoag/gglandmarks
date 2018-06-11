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
from keras.callbacks import TensorBoard
from time import time

class AbstractModel(object):
    def __init__(self, input_shape, num_classes, log_dir, weight_cache_folder=os.getcwd()):        
        self.weight_cache_folder = weight_cache_folder

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.create_model(input_shape, num_classes)        
        self.tensorboard = TensorBoard(log_dir= os.path.join(log_dir, self.name, time()))

    @property
    def name(self):
        raise NotImplementedError('Please implement the create model function')
    def create_model(self, input_shape, num_classes):
        raise NotImplementedError('Please implement the create model function')

    def summary(self):
        return self.model.summary()

    @property
    def model_weights_file(self):
        return os.path.join(self.weight_cache_folder, '{}_{}.h5'.format(self.name, self.num_classes))

    def train_and_validate(self, generator, epochs, validation_split, batch_size, train_steps=1, load_weight=True):
        """Train and validate model

        Arguments:
            generator [GoogleLandmarkDataset] -- Google landmark dataset to get train and validation generator
            epochs [int] -- The number of epochs
            validation_split [int] -- Validation split percentage (0..1)
            batch_size [int] -- Batch size

        Keyword Arguments:
            train_steps {int} -- The number of training steps before doing validation (default: {2})
            load_weight {bool} -- Continue training from the pre-train weight or not (default: {True})

        Returns:
            [train and evaluation results] -- [description]
        """

        gen = generator.get_train_validation_generator(
            batch_size, self.input_shape, validation_split)

        current_train_steps = 0
        if load_weight and os.path.exists(self.model_weights_file):
            self.model.load_weights(self.model_weights_file)

        train_history_metrics = []
        eval_history_metrics = []
        train_result_names = ['train_loss', 'train_accuracy']
        for i in range(epochs):
            train_gen, val_gen = next(gen)

            metrics = {}
            for name in self.model.metrics_names:
                metrics[name] = []

            for x, y in train_gen:
                print(x.shape)
                print(y.shape)
                train_results = self.model.train_on_batch(x, y)

                for idx, val in enumerate(train_results):
                    metrics[self.model.metrics_names[idx]] = val

                print('epoch {}/{} - train - metrics:{} '.format(i +
                                                                 1, epochs, train_results))

            train_history_metrics.append(metrics)

            self.model.save_weights(self.model_weights_file)

            current_train_steps += 1
            if current_train_steps == train_steps:

                for x, y in val_gen:
                    val_results = self.model.test_on_batch(x, y)

                    for idx, val in enumerate(val_results):
                        metrics[self.model.metrics_names[idx]] = val

                    print('epoch {}/{} - eval - metrics:{} '.format(i +
                                                                    1, epochs, train_results))

                eval_history_metrics.append(metrics)

        return train_history_metrics, eval_history_metrics

    def predict(self, X):
        return self.model.predict(X)