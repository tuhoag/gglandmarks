from gglandmarks.datasets import GoogleLandmarkDataset
import matplotlib.pyplot as plt
import pandas as pd

def record_stats(stats):
    print(stats)

def calculate_mean_and_std(df):
    # print(df.describe)
    new_df = df.groupby(by='landmark_id').count()
    # print(new_df)

    mean = new_df['path'].mean()
    # print('mean: {}'.format(mean))    

    std = new_df['path'].std()
    # print('std: {}'.format(std))

    maximum = new_df['path'].max()
    minimum = new_df['path'].min()

    return mean, std, maximum, minimum

def plot_stats(df):
    plt.plot(df['images_count_min'], df['num_classes'])
    plt.xlabel('The minimum number of images per class')
    plt.ylabel('The number of classes')
    plt.show()

def main(reload_stats=False):
    data_path = './data/landmarks_recognition/'
    image_original_size = (128, 128)
    images_count_min = 10000

    step = 50
    min_count = 0
    max_count = 10000

    if reload_stats:
        stats_df = pd.DataFrame(columns=['images_count_min', 'num_classes', 'num_examples', 'std', 'mean', 'max', 'min'])

        i = 0
        for images_count_min in range(min_count, max_count + 1, step):
            dataset = GoogleLandmarkDataset(
                data_path, (image_original_size[0], image_original_size[1]), images_count_min=images_count_min)

            mean, std, maximum, minimum = calculate_mean_and_std(dataset.train_df)

            stats = {
                'images_count_min': images_count_min,
                'num_classes': dataset.num_classes,
                'num_examples': dataset.train_df.shape[0],
                'std': std,
                'mean': mean,
                'max': maximum,
                'min': minimum
            }

            stats_df.loc[i] = stats
            i = i + 1
            print

    # plot_stats(stats_df)
    stat_path = './output/experiments/min_counts.csv'
    stats_df.to_csv(stat_path)

    plot_stats(pd.DataFrame.from_csv(stat_path))
    print(stats_df)