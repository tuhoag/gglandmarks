import pandas as pd
from PIL import Image
from io import BytesIO
from urllib import request, error
import os
import tensorflow as tf
import multiprocessing
from functools import reduce, partial

#import keras.preprocessing.image as krimage

DEFAULT_DATA_DIRECTORY='./data/landmarks_recognition/'
TRAIN_DEFAULT_DIRECTORY='./data/landmarks_recognition/'
TEST_DEFAULT_DIRECTORY = './data/landmarks_recognition/test/'
DEFAULT_ORIGINAL_IMG_DIRECTORY='original'
NUMBER_OF_SAMPLES = 5

def load_data(directory, resize):
    if resize is None:
        images_directory = DEFAULT_ORIGINAL_IMG_DIRECTORY
    else:
        images_directory = '{}_{}'.format(resize[0], resize[1])

    images_directory = os.path.join(directory, images_directory)
    train_directory = os.path.join(images_directory, 'train_index.csv')
    test_directory = os.path.join(images_directory, 'test_index.csv')

    train_df = pd.read_csv(train_directory)
    test_df = pd.read_csv(test_directory)

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
    print('loading dataframes')
    train_df_path = os.path.join(directory, 'train.csv')
    test_df_path= os.path.join(directory, 'test.csv')

    train_df = pd.read_csv(train_df_path, index_col=0)
    test_df = pd.read_csv(test_df_path, index_col=0)

    if sample:
        train_df = train_df.head(NUMBER_OF_SAMPLES)
        test_df = test_df.head(NUMBER_OF_SAMPLES)

    print('downloading data')
    if force_download:
        # download train
        download(train_directory, train_df, resize)
        # download test
        download(test_directory, test_df, resize)

    # build indexed files for train & test (remove missing data)
    print('building index files')
    indexed_train_df_path = os.path.join(images_directory, 'train_index.csv')
    indexed_test_df_path = os.path.join(images_directory, 'test_index.csv')

    indexed_train_df = build_indexed_file(train_directory, train_df, indexed_train_df_path)
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
    img_series = img_series.rename('path')

    data['path'] = img_series
    new_data = data.drop(columns=['url'])
    new_data.to_csv(indexed_path)

    return new_data

def check_chunk(chunk, directory):
    chunk_index, data = chunk
    print('start checking chunk: {}'.format(chunk_index))

    total_images = len(data)
    count = 0
    image_paths = {}

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
        print('[{}] - checked {} / {} image - {:4.4f}%'.format(chunk_index,
                                                                count, total_images, count/total_images*100))

    return image_paths

def split_chunks(data, num_processes):
    total_images = data.shape[0]
    chunk_size = int(total_images / num_processes) if total_images > num_processes else 1
    chunks_data = [data.iloc[i:i + chunk_size] for i in range(0, total_images, chunk_size)]
    chunks = [(i, chunks_data[i]) for i in range(len(chunks_data))]

    print('number of processes: {}'.format(num_processes))
    print('total number of images: {:,}'.format(total_images))
    print('chunk size: {:,}'.format(chunk_size))
    print('total chunks: {}'.format(len(chunks)))
    # print('data: {}'.format(data))
    # print('first chunk: {}'.format(chunks[0]))

    sum_chunk = 0
    for i in range(0, len(chunks)):
        print('chunk {} length: {}'.format(chunks[i][0], len(chunks[i][1])))
        sum_chunk += len(chunks[i][1])
    # sum_chunk = reduce(lambda x, acc: acc + len(x), chunks)
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
    img_series = img_series.rename('path')

    return img_series

def download_chunk(chunk, directory, resize):
    chunk_index, data = chunk
    print('start downloading chunk: {}'.format(chunk_index))
    total_images = len(data)
    count = 0
    image_paths = {}

    for index, row in data.iterrows():
        cls = None if 'landmark_id' not in data.columns else str(row['landmark_id'])
        image_path = download_image(row['url'], directory, index, resize, cls)
        if image_path != '':
            image_paths[index] = image_path

        count += 1
        print('[{}] - download {} / {} image - {:4.4f}%'.format(chunk_index, count, total_images, count/total_images*100))

    return image_paths

def download_image(url, directory, name, resize, cls):
    image_directory = directory if cls is None else os.path.join(directory, cls)
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    image_path = os.path.join(image_directory, '{}.jpg'.format(name))

    try:
        if os.path.exists(image_path):
            print('Image {} already exists. Skipping download.'.format(image_path))
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
        print('Warning: Could not download image {} from {}'.format(name, url))
        return ''
    except IOError as err:
        print('Warning: Failed to save image {}'.format(image_path))
        return ''

if '__main__' == __name__:
    train_df, test_df = init_dataset(directory=DEFAULT_DATA_DIRECTORY, sample=True, resize=(128, 128), force_download=False)
