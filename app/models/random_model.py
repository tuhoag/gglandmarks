from sklearn.dummy import DummyClassifier
from sklearn.utils import check_X_y, check_array
import pandas as pd
import numpy as np

class RandomModel(object):
    def __init__(self):
        self.model = DummyClassifier(strategy='stratified', random_state=0)
        pass

    def from_train_data(self, df):
        X = np.array(df.index.tolist()).reshape(-1, 1)
        Y = np.array(df['landmark_id'].tolist()).reshape(-1, 1)
        return check_X_y(X=X, y=Y)

    def from_test_data(self, df):
        X = np.array(df.index.tolist()).reshape(-1, 1)

        return check_array(X)


    def fit(self, train_df):
        X, Y = self.from_train_data(train_df)
        self.model.fit(X, Y)

    def predict(self, test_df):
        """
            Predict labels of a set of images
        Arguments:
            X {ndarray} -- array of images (n * width * height * channel)
        """
        X = self.from_test_data(test_df)
        predictions = self.model.predict(X)
        # print(predictions)
        predictions_probs = self.model.predict_proba(X)

        return predictions, predictions_probs

    def generate_submission(self, test_df, filename):
        predictions, probs = self.predict(test_df)
        print(predictions)
        print(probs)
        with(open(filename, 'w')) as f:
            f.write('id,landmarks\n')
            i = 0
            for _, row in test_df.iterrows():
                f.write('{}, {} {}\n'.format(row['id'], predictions[i], probs[i][predictions[i]]))
                i += 1

