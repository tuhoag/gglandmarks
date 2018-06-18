import pandas as pd
from PIL import Image
from io import BytesIO
from urllib import request, error
import os
import tensorflow as tf
import multiprocessing
from functools import reduce, partial
from tqdm import tqdm
from time import sleep
import math
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import Sequence
import keras
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import time
from .data_generator import DataGenerator
from .encoder import Encoder

DEFAULT_DATA_DIRECTORY='./data/landmarks_recognition/'
TRAIN_DEFAULT_DIRECTORY='./data/landmarks_recognition/'
TEST_DEFAULT_DIRECTORY = './data/landmarks_recognition/test/'
DEFAULT_ORIGINAL_IMG_DIRECTORY='original'
NUMBER_OF_SAMPLES = 1000

class GoogleLandmarkDataset(object):
    def __init__(self, directory, size, images_count_min=None, is_clean_data=True):
        self.train_df, self.test_df = load_data(directory, size)
        self.size = size
        self.images_count_min = images_count_min

        print('train before clean: {}'.format(self.train_df.shape))
        if is_clean_data:
            self.train_df = self.clean_train_data(self.train_df, self.images_count_min)
            print('train after clean: {}'.format(self.train_df.shape))

        # build classes indexes
        self.encoder = self.create_label_indexes(self.train_df['landmark_id'])

    def clean_train_data(self, train_df, images_count_min):
        """Remove classes have less than images_count_min data

        Arguments:
            train_df [DataFrame] -- train data
            images_count_min [int] -- minimum images per class

        Returns:
            [DataFrame] -- The cleaned training data
        """

        stats = self.get_frequent_landmarks(train_df, images_count_min)
        landmarks = stats.index.tolist()
        # print('frequent landmarks len: {}'.format(len(landmarks)))
        print('train before remove: {}'.format(train_df.shape))
        # print(landmarks)
        # print(train_df['landmark_id'].describe())
        # print(train_df['landmark_id'])
        # print(train_df['landmark_id'].isin(landmarks))
        # print(train_df[train_df['landmark_id'].isin(landmarks)])
        new_train_df = train_df[train_df['landmark_id'].isin(landmarks)]
        print('train after remove: {}'.format(new_train_df.shape))
        new_train_df = new_train_df[new_train_df['path'] != '']

        # new_train_df.to_csv('temp.csv')

        return new_train_df

    def get_frequent_landmarks(self, train_df, images_count_min):
        """Get all landmarks have the number of images higher than images_count_min

        Arguments:
            train_df [DataFrame] -- Training data
            images_count_min [int] -- The minium number of images per landmark

        Returns:
            [dictionary] -- landmark and the number of images per landmark
        """

        train_df = train_df[train_df['path'] != '']

        stats = train_df.groupby(['landmark_id']).count().sort_values(by=['path'])

        if not images_count_min is None:
            stats = stats[stats['path'] >= images_count_min]

        return stats

    @property
    def num_classes(self):
        return self.encoder.classes_.shape[0]

    @property
    def classes(self):
        return self.encoder.classes_

    def create_label_indexes(self, landmarks):
        """Create scikit mapping from landmarks to id

        Arguments:
            landmarks [Panda Series] -- All landmarks in dataset

        Returns:
            encoder -- LabelBinarizer mapping from landmark to index
        """

        encoder = Encoder(landmarks)
        # print(len(encoder.classes_))
        return encoder

    def get_train_validation_generator(self, batch_size, target_size, validation_size, output_type='generator'):
        """Get train and validation generators

        Arguments:
            batch_size [int] -- Batch size
            target_size [tuple(width, height)] -- Tuple of image width and height
            validation_size [type] -- Percentage of validation set
            output_type [string] -- 'generator' or 'dataset'. The former is normal Python generator while the latter is the tf dataset

        Returns:
            (DataGenerator, DataGenerator) -- Train and validation generator
        """


        train_df = self.train_df

        paths = train_df['path'].tolist()
        landmarks = train_df['landmark_id'].tolist()

        while(True):
            train_path, test_path, train_landmark, test_landmark = train_test_split(paths, landmarks, test_size=validation_size)
            print('finish splitting train & validation')
            data_gen = self.data_generator(batch_size, target_size, output_type)

            yield data_gen(train_path, train_landmark), data_gen(test_path, test_landmark)

    def data_generator(self, batch_size, target_size, output_type):
        def create_data_generator(paths, landmarks):
            return DataGenerator(paths, landmarks, self.encoder, batch_size, target_size)

        def create_data_set(paths, landmarks):
            labels = self.encoder.class_to_label(landmarks)
            dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
            dataset = dataset.map(_parse)
            dataset = dataset.shuffle(batch_size)
            dataset = dataset.batch(batch_size)            
            dataset = dataset.prefetch(batch_size)
            # dataset = tf.data.Dataset.from_tensor_slices((paths, labels)).map(_parse).batch(batch_size).prefetch(batch_size)

            return dataset

        def _parse(image_path, label):
            print(image_path)

            image_string = tf.read_file(image_path)
            # image_decoded = tf.image.decode_image(image_string, channels=3)
            image_decoded = tf.cond(
                tf.image.is_jpeg(image_string),
                lambda: tf.image.decode_jpeg(image_string, channels=3),
                lambda: tf.image.decode_png(image_string, channels=3))

            image_resized = tf.image.resize_images(image_decoded, target_size)

            # one_hot_label = tf.py_func(_encode, [landmark], tf.int64)
            # return image_decoded, landmark
            return {'image': image_resized}, label

        def _encode(landmark):
            return np.array([0, 0, 1])

        if output_type == 'generator':
            return create_data_generator
        elif output_type == 'dataset':
            return create_data_set
        else:
            raise AttributeError('Output type must be generator or dataset')


class GoogleLandmarkTestGenerator(Sequence):
    def __init__(self, directory, size, batch_size, target_size):
        self.directory = directory,
        self.size = size
        self.batch_size = batch_size
        self.target_size = target_size
        df_path = os.path.join(get_image_path(directory, size), 'test_index.csv')

        if not os.path.exists(df_path):
            raise FileNotFoundError('not found test indexed data')
        df = pd.read_csv(df_path, keep_default_na=False)
        self.data = df[df['path'] != '']

        print('total images ', len(df))
        print('total nonempty image ', len(self.data))

    def __len__(self):
        return math.ceil(len(self.data) / float(self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        batch_data = self.data[start_idx:end_idx]

        images_batch = np.empty((0, self.target_size[0], self.target_size[1], 3))
        idx_batch = np.empty((0))

        for _, row in batch_data.iterrows():
            image_array = load_image(row['path'], self.target_size)
            if image_array is not None:
                image_array = image_array.reshape(1, self.target_size[0], self.target_size[1], 3)
                images_batch = np.append(images_batch, image_array, axis=0)
                idx_batch = np.append(idx_batch, [row['id']])

        return idx_batch, images_batch

    def on_epoch_end(self):
        print('end epoch')

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

def load_missing_test_data(directory, size):
    train_df, test_df = load_data(directory, size)

    # remove '' records
    print(len(test_df))
    test_df = test_df[test_df['path'] == '']
    print(len(test_df))

    return test_df

def load_image(image_path, target_size):
    # print(image_path)
    # try:
    # if image_path != '':
        image = load_img(image_path, target_size=target_size)
        image_array = img_to_array(image)
        return image_array

    # return None
    # except:
    #     print(image_path)
    #     return None

    # print(image_array.shape)


def load_chunk_images(chunk, directory, resize):
    chunk_index, data = chunk
    total_images = len(data)
    image_paths = np.empty((1, 128, 128, 3), int)

    pbar = tqdm(desc='chunk {}'.format(chunk_index), total=total_images, position=chunk_index)

    for index, row in data.iterrows():

        image_array = load_image(row['path'])
        if image_array is not None:
            image_array = image_array.reshape(1, 128, 128, 3)
            image_paths = np.vstack((image_paths, image_array))

        pbar.update(1)

    pbar.close()
    return image_paths

def load_test_images(directory, resize):
    train_df, test_df = load_data(directory, resize)

    # remove '' records
    print(len(test_df))
    test_df = test_df[test_df['path'] != '']
    print(len(test_df))
    num_processes = multiprocessing.cpu_count()
    chunks = split_chunks(test_df, num_processes)

    pool = multiprocessing.Pool(processes=num_processes)
    prod = partial(load_chunk_images, directory=directory, resize=resize)
    images = np.array(pool.map(prod, chunks))
    new_images = images.reshape(-1, 128, 128, 3)

    assert(np.array_equal(images[0][0],new_images[0]))
    print(images.shape)

    return images

def get_image_path(directory, resize):
    if resize is None:
        images_directory = DEFAULT_ORIGINAL_IMG_DIRECTORY
    else:
        images_directory = '{}_{}'.format(resize[0], resize[1])

    images_directory = os.path.join(directory, images_directory)
    return images_directory


def load_data(directory, resize):
    images_directory = get_image_path(directory, resize)
    train_directory = os.path.join(images_directory, 'train_index.csv')
    test_directory = os.path.join(images_directory, 'test_index.csv')

    train_df = pd.read_csv(train_directory, keep_default_na=False)
    test_df = pd.read_csv(test_directory, keep_default_na=False)

    return train_df, test_df

def load_raw_data(directory):
    train_directory = os.path.join(directory, 'train.csv')
    test_directory = os.path.join(directory, 'test.csv')

    train_df = pd.read_csv(train_directory)
    test_df = pd.read_csv(test_directory)

    return train_df, test_df

def init_dataset(directory=DEFAULT_DATA_DIRECTORY, sample = False, resize=None, force_download=True):
    # check and create data folders
    print('initing dataset')
    print('creating folders')
    if resize is None:
        images_directory = DEFAULT_ORIGINAL_IMG_DIRECTORY
    else:
        images_directory = '{}_{}'.format(resize[0], resize[1])

    images_directory = os.path.join(directory, images_directory)
    train_directory = os.path.join(images_directory, 'train')
    test_directory = os.path.join(images_directory, 'test')

    directories = [images_directory, train_directory, test_directory]
    for each in directories:
        if not os.path.exists(each):
            os.makedirs(each)

    # load original train & test data
    print('LOADING DATAFRAMES')
    train_df_path = os.path.join(directory, 'train.csv')
    test_df_path= os.path.join(directory, 'test.csv')

    train_df = pd.read_csv(train_df_path, index_col=0)
    test_df = pd.read_csv(test_df_path, index_col=0)

    if sample:
        train_df = train_df.head(NUMBER_OF_SAMPLES)
        test_df = test_df.head(NUMBER_OF_SAMPLES)

    print('DOWNLOADING DATA')
    if force_download:
        # download train
        print('DOWNLOADING TRAIN DATA')
        download(train_directory, train_df, resize)
        # download test
        print('DOWNLOADING TEST DATA')
        download(test_directory, test_df, resize)

    # build indexed files for train & test (remove missing data)
    print('BUILDING INDEX FILES')
    indexed_train_df_path = os.path.join(images_directory, 'train_index.csv')
    indexed_test_df_path = os.path.join(images_directory, 'test_index.csv')

    print('BUILDING TRAIN INDEX FILE')
    indexed_train_df = build_indexed_file(train_directory, train_df, indexed_train_df_path)

    print('BUILDING TEST INDEX FILE')
    indexed_test_df = build_indexed_file(test_directory, test_df, indexed_test_df_path)

    return indexed_train_df, indexed_test_df

def build_indexed_file(images_directory, data, indexed_path):
    num_processes = multiprocessing.cpu_count()
    chunks = split_chunks(data, num_processes)

    pool = multiprocessing.Pool(processes=num_processes)
    prod = partial(check_chunk, directory=images_directory)
    image_paths = pool.map(prod, chunks)

    img_series = pd.Series()

    for each in image_paths:
        img_series = img_series.append(pd.Series(each))
        print('')
    img_series = img_series.rename('path')

    data['path'] = img_series
    new_data = data.drop(columns=['url'])
    new_data.to_csv(indexed_path)

    return new_data

def check_chunk(chunk, directory):
    chunk_index, data = chunk
    # print('start checking chunk: {}'.format(chunk_index))

    total_images = len(data)
    count = 0
    image_paths = {}

    # tqdm.pandas(desc='{}'.format(chunk_index))
    pbar = tqdm(desc='chunk {}'.format(chunk_index), total=total_images, position=chunk_index)
    # with tqdm(total=total_images, position=chunk_index) as pbar:
    for index, row in data.iterrows():
        cls = row.get('landmark_id')
        if cls is None:
            image_path = os.path.join(directory, '{}.jpg'.format(index))
        else:
            image_path = os.path.join(directory, str(cls), '{}.jpg'.format(index))

        if os.path.exists(image_path):
            image_paths[index] = image_path
        else:
            image_paths[index] = ''

        count += 1
        # data.progress_apply(lambda x: x + 1)
        pbar.update(1)
            # print('[{}] - checked {} / {} image - {:4.4f}%'.format(chunk_index,
                                                                    # count, total_images, count/total_images*100))

    pbar.close()
    return image_paths

def split_chunks(data, num_processes):
    total_images = data.shape[0]
    chunk_size = math.ceil(total_images / num_processes) if total_images > num_processes else 1
    chunks_data = [data.iloc[i:i + chunk_size] for i in range(0, total_images, chunk_size)]
    chunks = [(i, chunks_data[i]) for i in range(len(chunks_data))]

    print('number of processes: {}'.format(num_processes))
    print('total number of images: {:,}'.format(total_images))
    print('chunk size: {:,}'.format(chunk_size))
    print('total chunks: {}'.format(len(chunks)))

    sum_chunk = 0
    for i in range(0, len(chunks)):
        print('chunk {} length: {}'.format(chunks[i][0], len(chunks[i][1])))
        sum_chunk += len(chunks[i][1])

    print('sum chunks: {:,}'.format(sum_chunk))

    assert(sum_chunk == data.shape[0])

    return chunks

def download(directory, data, resize):
    num_processes = multiprocessing.cpu_count()
    chunks = split_chunks(data, num_processes)

    pool = multiprocessing.Pool(processes=num_processes)
    prod = partial(download_chunk, directory=directory, resize=resize)
    image_paths = pool.map(prod, chunks)
    img_series = pd.Series()

    for each in image_paths:
        img_series = img_series.append(pd.Series(each))
        print('')
    img_series = img_series.rename('path')

    return img_series

def download_chunk(chunk, directory, resize):
    chunk_index, data = chunk
    total_images = len(data)
    image_paths = {}

    pbar = tqdm(desc='chunk {}'.format(chunk_index), total=total_images, position=chunk_index)
    for index, row in data.iterrows():
        cls = None if 'landmark_id' not in data.columns else str(row['landmark_id'])
        image_path = download_image(row['url'], directory, index, resize, cls)
        if image_path != '':
            image_paths[index] = image_path

        pbar.update(1)

    pbar.close()
    return image_paths

def download_image(url, directory, name, resize, cls):
    image_directory = directory if cls is None else os.path.join(directory, cls)
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    image_path = os.path.join(image_directory, '{}.jpg'.format(name))

    try:
        if os.path.exists(image_path):
            return image_path

        response = request.urlopen(url)
        image_data = response.read()
        pil_image = Image.open(BytesIO(image_data))
        pil_image_rgb = pil_image.convert('RGB')

        if resize is not None:
            pil_image_rgb = pil_image_rgb.resize(resize)

        pil_image_rgb.save(image_path, format='JPEG', quality=90)

        return image_path
    except error.URLError as err:
        # print('Warning: Could not download image {} from {}'.format(name, url))
        return ''
    except IOError as err:
        # print('Warning: Failed to save image {}'.format(image_path))
        return ''

if '__main__' == __name__:
    train_df, test_df = init_dataset(directory=DEFAULT_DATA_DIRECTORY, sample=False, resize=(128, 128), force_download=True)
