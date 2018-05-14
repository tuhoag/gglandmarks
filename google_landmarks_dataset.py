import pandas as pd
from PIL import Image
from io import BytesIO
from urllib import request, error
import os
import tensorflow as tf
import multiprocessing
from functools import reduce, partial

#import keras.preprocessing.image as krimage

DEFAULT_DIRECTORY='./data/landmarks_recognition/'
DEFAULT_ORIGINAL_IMG_DIRECTORY='original'

def load_data(directory=DEFAULT_DIRECTORY, force_download=True):
    train = dataset(directory, 'train.csv', 'images', force_download)
    test = pd.read_csv(directory + 'test.csv')

    return train, test

def load_train_dataset(directory=DEFAULT_DIRECTORY, force_download=True):
    return dataset(directory, 'train.csv', DEFAULT_ORIGINAL_IMG_DIRECTORY, force_download)

def load_test_dataset(directory=DEFAULT_DIRECTORY, force_download=True):
    return dataset(directory, 'test.csv', DEFAULT_ORIGINAL_IMG_DIRECTORY, force_download)

def dataset(directory, data_file, images_folder, force_download=True):
    df = pd.read_csv(os.path.join(directory, data_file))
    images = load_images(os.path.join(directory, images_folder), df, force_download)

    return images, df['landmark_id'].values

def load_images(directory, data, force_download):
    if force_download:
        image_paths = download(directory, data)
        # remove empty

    images = load_local_images(directory, data)

    return image_paths, images

def load_local_images(directory, data):
    images = []
    for _, row in data.iterrows():
        image_path = row['path']
        images.append(tf.keras.preprocssing.image.load_img(image_path))

    return images

def split_chunks(data, num_processes):
    total_images = data.shape[0]
    chunk_size = int(total_images / num_processes)
    chunks_data = [data.iloc[i:i + chunk_size] for i in range(0, total_images, chunk_size)]
    chunks = [(i, chunks_data[i]) for i in range(len(chunks_data))]

    print('number of processes: {}'.format(num_processes))
    print('total number of images: {:,}'.format(total_images))
    print('chunk size: {:,}'.format(chunk_size))
    print('total chunks: {}'.format(len(chunks)))
    print('data: {}'.format(data))
    print('first chunk: {}'.format(chunks[0]))
    
    sum_chunk = 0
    for i in range(0, len(chunks)):
        print('chunk {} length: {}'.format(chunks[i][0], len(chunks[i][1])))
        sum_chunk += len(chunks[i][1])
    # sum_chunk = reduce(lambda x, acc: acc + len(x), chunks)    
    print('sum chunks: {:,}'.format(sum_chunk))

    assert(sum_chunk == data.shape[0])

    return chunks

def download(directory, data):
    num_processes = multiprocessing.cpu_count()
    chunks = split_chunks(data, num_processes)

    
    pool = multiprocessing.Pool(processes=num_processes)
    prod = partial(download_chunk, directory=directory)
    result = pool.map(prod, chunks)
    print(result)
    # image_paths = []
        

def download_chunk(chunk, directory):
    chunk_index, data = chunk
    print('start downloading chunk: {}'.format(chunk_index))
    total_images = len(data)
    count = 0
    image_paths = []

    for _, row in data.iterrows():
        image_path = download_image(row['url'], directory, row['id'])
        if image_path != '':
            image_paths.append({'id': row['id'], 'path': image_path})

        count += 1
        print('[{}] - download {} / {} image - {:4.4f}%'.format(chunk_index, count, total_images, count/total_images*100))

    return image_paths

def download_image(url, directory, name):
    image_path = os.path.join(directory, '{}.jpg'.format(name))

    try:
        if os.path.exists(image_path):
            print('Image {} already exists. Skipping download.'.format(image_path))
            return image_path

        response = request.urlopen(url)
        image_data = response.read()
        pil_image = Image.open(BytesIO(image_data))
        pil_image_rgb = pil_image.convert('RGB')
        pil_image_rgb.save(image_path, format='JPEG', quality=90)

        return image_path
    except error.URLError as err:
        print('Warning: Could not download image {} from {}'.format(name, url))
        return ''
    except IOError as err:
        print('Warning: Failed to save image {}'.format(image_path))
        return ''

def print_data_stats(data):
    series = pd.Series(data, name='data')
    print(series.describe())

if '__main__' == __name__:
    trainX, trainY = load_train_dataset()
    testX, testY = load_test_dataset()

    print('train stats')
    print_data_stats(trainY)
    print('test stats')
    print_data_stats(testY)