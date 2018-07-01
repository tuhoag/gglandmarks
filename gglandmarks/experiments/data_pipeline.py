from gglandmarks.datasets import GoogleLandmarkDataset
import time
import tensorflow as tf
import pandas as pd
# import cpu
import multiprocessing

def calculate_generator(dataset, trial, batch_size, target_size):
    gen = dataset.get_train_validation_generator(
        batch_size=batch_size, target_size=target_size, validation_size=0.1, output_type='generator')
    train_dataset, _ = next(gen)

    # print('batch_size:{}'.format(batch_size))
    total_duration = 0
    train_iter = iter(train_dataset)

    for i in range(trial):
        start_time = time.time()
        x, y = next(train_iter)
        end_time = time.time()

        duration = end_time - start_time
        total_duration += duration
        print('time: {}'.format(duration))

    avg_duration = total_duration / trial
    stats = {
        'type': 'generator',
        'batch_size': batch_size,
        'total_time': total_duration,
        'average': avg_duration,
        'trial': trial
    }

    return stats


def calculate_dataapi(dataset, trial, batch_size, target_size, num_parallel):
    if num_parallel == 1:
        num_parallel = None

    gen = dataset.get_train_validation_generator(
        batch_size=batch_size, target_size=target_size, validation_size=0.1, output_type='dataset', shuffle=False, num_parallel=num_parallel)
    train_dataset, _ = next(gen)
    iterator = train_dataset.make_initializable_iterator()
    features, labels = iterator.get_next()

    total_duration = 0
    print('calculating dataapi for:{}'.format(num_parallel))

    with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer)
        sess.run(iterator.initializer)
        for i in range(trial):
            start_time = time.time()
            x, y = sess.run([features, labels])
                # print(y)
            end_time = time.time()

            duration = end_time - start_time
            total_duration += duration
            print('time: {}'.format(duration))

        avg_duration = total_duration / trial

        stats = {
            'type': 'dataapi-{}'.format(num_parallel),
            'batch_size': batch_size,
            'total_time': total_duration,
            'average': avg_duration,
            'trial': trial
        }

        return stats


def load_data():
    trial = 10

    max_batch_size = 1024
    cpu_counts = [ i for i in range(2, multiprocessing.cpu_count() + 1, 2)]
    cpu_counts.insert(0, 1)
    print(cpu_counts)
    batch_sizes = [2 ** i for i in range(5, 11)]
    print(batch_sizes)
    data_path = './data/landmarks_recognition/'
    image_original_size = (128, 128)
    target_size = (128, 128)
    dataset = GoogleLandmarkDataset(
        data_path, (image_original_size[0], image_original_size[1]), images_count_min=None)

    stats_df = pd.DataFrame(
        columns=['type', 'batch_size', 'total_time', 'average', 'trial'])
    index = 0

    cpu_counts = list(range(1, multiprocessing.cpu_count()))

    for batch_size in batch_sizes:
        print('batch size:{}'.format(batch_size))
        stats = calculate_generator(dataset, trial, batch_size, target_size)
        stats_df.loc[index] = stats
        index += 1
        print('python generator: {}'.format(stats))

        for num_cpus in cpu_counts:
            stats = calculate_dataapi(dataset, trial, batch_size, target_size, num_cpus)
            stats_df.loc[index] = stats
            index += 1
            print('tf dataapi: {}'.format(stats))

    print(stats_df)
    return stats_df

def main(reload_stats=True):
    if reload_stats:
        stats_df1 = load_data()
        stats_df1.to_csv('./output/experiments/loading_time.csv')

