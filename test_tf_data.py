from gglandmarks.datasets import GoogleLandmarkDataset
import os
import tensorflow as tf

# load data
image_path = './data/landmarks_recognition/'
image_train_path = os.path.join(image_path, 'train')
image_test_path = os.path.join(image_path, 'test')
image_archive_test_path = os.path.join(image_path, 'test1')
num_classes = 14951
batch_size = 64
epochs = 10
target_size = (64, 64)
# create_train_and_test_generators(image_path, (128,128))
dataset = GoogleLandmarkDataset(
    image_path, (128, 128), images_count_min=1000)

dataset_generator = dataset.get_train_validation_generator(
    batch_size=32,
    target_size=target_size,
    validation_size=0.1,
    output_type='dataset')

train_dataset, val_dataset = next(dataset_generator)

print(train_dataset)
print(val_dataset)
print(dataset.num_classes)
print(dataset.classes)

train_iter = train_dataset.make_initializable_iterator()

X, Y = train_iter.get_next()

with tf.Session() as sess:
    sess.run(train_iter.initializer)

    # while True:
    X_val, Y_val = sess.run([X, Y])
    print(X_val.shape)
    print(Y_val.shape)
