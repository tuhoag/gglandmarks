import tensorflow as tf
import datetime
import os

class TFBaseModel():
    def __init__(self, name, model_dir, model_fn):
        self.name = name,
        self.model_dir = model_dir
        self.model_fn = model_fn

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

    # def build(self, model_fn, dataset, batch_size, target_size, params):
    def build(self, dataset, batch_size, target_size, params):
        tf.reset_default_graph()
        features, labels = self.import_data(dataset, batch_size, target_size)

        self.global_step = tf.get_variable(
            name='global_step', trainable=False, initializer=tf.constant(0))

        self.spec = self.model_fn(features, labels, mode=None, params=params)

    def train_one_epoch(self, input_fn, writer, current_step, session, steps=None):
        # train
        train_init = input_fn()
        loss = self.spec.loss
        train_op = self.spec.train_op
        merged_summary = tf.summary.merge_all()

        session.run(train_init)

        step = 0
        total_loss = 0
        try:
            while True:
                step += 1
                train_loss, _, s, current_step = session.run(
                    [loss, train_op, merged_summary, self.global_step])
                writer.add_summary(s, current_step)
                print('current step: {}'.format(current_step))
                print('{} - train loss: {}'.format(step, train_loss))
                total_loss += train_loss

                if(steps is not None and step >= steps):
                    break

        except tf.errors.OutOfRangeError as err:
            print('end epoch:')

        return total_loss, current_step

    def evaluate(self, input_fn, writer, current_step, session, steps=None):
        merged_summary = tf.summary.merge_all()
        eval_init = input_fn()
        metrics_ops = self.spec.eval_metric_ops

        session.run(eval_init)

        step = 0
        total_accuracy = 0
        try:
            while True:
                step += 1
                s, metrics = session.run([merged_summary, metrics_ops])
                writer.add_summary(s, current_step)
                print('current step: {}'.format(current_step))
                print('metrics: {}'.format(metrics))
                total_accuracy += metrics['accuracy'][1]

                if(steps is not None and step >= steps):
                    break

        except tf.errors.OutOfRangeError as err:
            print('end epoch')

        return total_accuracy

    def fit(self, train_iter, eval_iter, logname, epochs=1000, steps=10000):
        """
        """
        writer_path = os.path.join(
            self.model_dir, logname + '-' + str(datetime.datetime.now()))
        train_writer = tf.summary.FileWriter(writer_path + '-train')
        eval_writer = tf.summary.FileWriter(writer_path + '-eval')

        current_step = 0
        total_losses = []
        total_accuracies = []

        # self.build(dataset, 50, self.image_shape, self.num_classes, 0.001)
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            train_writer.add_graph(sess.graph)
            eval_writer.add_graph(sess.graph)

            for i in range(epochs):
                print("Traing epoch:{}".format(i))
                total_loss, current_step = self.train_one_epoch(
                    lambda: train_iter, current_step=current_step, writer=train_writer, steps=steps, session=sess)

                total_losses.append(total_loss)

                if i % 10 == 0:
                    print("Evaluating epoch: {}".format(i))
                    total_accuracy = self.evaluate(
                        lambda: eval_iter, current_step=current_step, writer=eval_writer, session=sess, steps=steps)

                total_accuracies.append(total_accuracy)

            return total_losses, total_accuracies