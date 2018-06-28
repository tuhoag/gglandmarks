import tensorflow as tf
import datetime
import os
import random

def _optimize(loss, params):
    with tf.variable_scope('train'): 
        global_step = tf.train.get_global_step()

        if 'decay_steps' in params:
            print('learning rate decay')
            learning_rate = tf.train.exponential_decay(params['learning_rate'], global_step,
                                           params['decay_steps'], 0.96, staircase=True)
        else:
            print('normal learning rate')
            learning_rate = tf.constant(params['learning_rate'])
        
        tf.summary.scalar('learning_rate', learning_rate)
        # tf.summary.scalar('global_step', global_step)
        # Passing global_step to minimize() will increment it at each step.
        train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(loss, global_step=global_step)

        return train_op

class TFBaseModel():
    def __init__(self, name, model_dir, model_fn):
        self.name = name
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

    def train_one_epoch(self, input_fn, writer, saver, save_path, session, steps=None):
        # train
        save_delay_steps = 100
        logs_delay_steps = 100

        train_init = input_fn()
        loss = self.spec.loss
        train_op = self.spec.train_op
        merged_summary = tf.summary.merge_all()

        session.run(train_init)
        step = 0
        total_loss = 0
        try:
            while True:    
                if(steps is not None and step >= steps):
                    break

                train_loss, _, current_step = session.run(
                    [loss, train_op, self.global_step])
                print('{} - train loss: {}'.format(current_step, train_loss))
                total_loss += train_loss

                if current_step % save_delay_steps == 0:
                    # save
                    saver_path = saver.save(session, save_path, global_step=self.global_step)
                    print('save model to: {}'.format(saver_path))

                if current_step % logs_delay_steps == 0:
                    # log
                    s = session.run(merged_summary)
                    writer.add_summary(s, current_step)
                    print('write log')

                step += 1

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
                if(steps is not None and step >= steps):
                    break
                
                s, metrics = session.run([merged_summary, metrics_ops])
                # writer.add_summary(s, current_step)
                # print('write log')
                print('evaluate current step: {} / {}, current_step: {}'.format(step, steps, current_step))
                print('metrics: {}'.format(metrics))
                total_accuracy += metrics['accuracy'][1]

                step += 1

        except tf.errors.OutOfRangeError as err:
            print('end epoch')        

        # Take the mean of you measure
        writer.add_summary(s, current_step)
        print('write log')
        accuracy = total_accuracy / step

        return accuracy

    def fit(self, train_iter, eval_iter, logname, epochs=100, steps=1000):
        """
        """
        writer_path = os.path.join(
            self.model_dir, self.name, logname)
        train_writer = tf.summary.FileWriter(writer_path + '-train')
        eval_writer = tf.summary.FileWriter(writer_path + '-eval')
        save_path = os.path.join(writer_path, 'weights')
        print(self.model_dir)
        print(writer_path)
        print(save_path)
        current_step = 0
        total_losses = []
        total_accuracies = []
        saver = tf.train.Saver()

        # self.build(dataset, 50, self.image_shape, self.num_classes, 0.001)
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            train_writer.add_graph(sess.graph)
            eval_writer.add_graph(sess.graph)

            # cpst = tf.train.get_checkpoint_state(writer_path)
            # if cpst is not None:
            last_checkpoint = tf.train.latest_checkpoint(os.path.dirname(save_path))
            print(last_checkpoint)
            if last_checkpoint is not None:
                saver.restore(sess, last_checkpoint)
                print('model restored from: {}'.format(last_checkpoint))

            for i in range(epochs):
                print("Traing epoch:{}".format(i))
                total_loss, current_step = self.train_one_epoch(
                    lambda: train_iter, writer=train_writer, saver=saver, save_path=save_path, steps=steps, session=sess)

                total_losses.append(total_loss)

                print("Evaluating epoch: {}".format(i))
                total_accuracy = self.evaluate(
                    lambda: eval_iter, current_step=current_step, writer=eval_writer, session=sess, steps=steps)

                total_accuracies.append(total_accuracy)

            return total_losses, total_accuracies
