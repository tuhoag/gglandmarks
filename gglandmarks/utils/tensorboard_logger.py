import tensorflow as tf

class TensorboardLogger(object):
    """This is a tensorboard logger. This class is a modified version of the
    logger here: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    """

    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, name, value, step):
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=value)

        self.writer.add_summary(summary, step)