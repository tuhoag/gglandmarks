from random_model import RandomModel
from google_landmarks_dataset import load_meta_data, DEFAULT_DATA_DIRECTORY, load_raw_data

if __name__ == "__main__":
    model = RandomModel()
    # train_df, test_df = load_data(directory=DEFAULT_DATA_DIRECTORY, resize=(128, 128))
    train_df, test_df = load_raw_data(directory=DEFAULT_DATA_DIRECTORY)
    model.fit(train_df)
    model.generate_submission(test_df, 'baseline_submission.csv')