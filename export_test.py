import google_landmarks_dataset
import numpy as np
DEFAULT_DATA_DIRECTORY='./data/landmarks_recognition/'
test_images = google_landmarks_dataset.load_test_images(DEFAULT_DATA_DIRECTORY, (128, 128))

np.savez(DEFAULT_DATA_DIRECTORY + 'test_128_128', test_images)
np.save(DEFAULT_DATA_DIRECTORY + 'test_128_128', test_images)