from gglandmarks.datasets import GoogleLandmarkDataset
import time
import tensorflow as tf
import pandas as pd


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
    


def calculate_dataapi(dataset, trial, batch_size, target_size):
    gen = dataset.get_train_validation_generator(
            batch_size=batch_size, target_size=target_size, validation_size=0.1, output_type='dataset', shuffle=False)
    train_dataset, _ = next(gen)
    iterator = train_dataset.make_initializable_iterator()
    features, labels = iterator.get_next()

    total_duration = 0

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
            'type': 'dataapi',
            'batch_size': batch_size,
            'total_time': total_duration,
            'average': avg_duration,
            'trial': trial
        }

        return stats


def load_data():
    epochs = 10
    max_batch_size = 1000
    data_path = './data/landmarks_recognition/'
    image_original_size = (128, 128)
    target_size = (128, 128)
    dataset = GoogleLandmarkDataset(
        data_path, (image_original_size[0], image_original_size[1]), images_count_min=None)

    stats_df = pd.DataFrame(
        columns=['type', 'batch_size', 'total_time', 'average', 'trial'])
    index = 0
    for batch_size in range(50, max_batch_size, 50):
        print('batch size:{}'.format(batch_size))
        stats = calculate_generator(dataset, epochs, batch_size, target_size)        
        stats_df.loc[index] = stats        
        index += 1
        print(stats)

        stats = calculate_dataapi(dataset, epochs, batch_size, target_size)
        stats_df.loc[index] = stats
        index += 1
        print(stats)

    print(stats_df)
    return stats_df

def main(reload_stats=True):
    if reload_stats:
        stats_df1 = load_data()
        stats_df1.to_csv('./output/experiments/loading_time.csv')

