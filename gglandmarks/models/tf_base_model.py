import tensorflow as tf

class TFBaseModel():
    def __init__(self, image_shape, num_classes):
        self.image_shape = image_shape
        self.num_classes = num_classes

    def import_data(self, dataset, batch_size, target_size, validation_split=0.1, output_type='dataset'):
        dataset_generator = dataset.get_train_validation_generator(
            batch_size=batch_size,
            target_size=target_size,
            validation_size=validation_split,
            output_type=output_type)

        train_dataset, val_dataset = next(dataset_generator)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                   train_dataset.output_shapes)

        features, labels = iterator.get_next()
        # self.X = features['image']

        self.train_iter = iterator.make_initializer(train_dataset)
        self.eval_iter = iterator.make_initializer(val_dataset)

        return features, labels

    def build(self, model_fn, dataset, batch_size, target_size, params):
        features, labels = self.import_data(dataset, batch_size, target_size)

        self.global_step = tf.get_variable(
            name='global_step', trainable=False, initializer=tf.constant(0))

        self.spec = model_fn(features, labels, mode=None, params=params)