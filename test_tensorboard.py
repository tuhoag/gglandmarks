import tensorflow as tf
from gglandmarks.utils import write_log
from keras.callbacks import TensorBoard
import os
from time import time

log_dir = os.path.join('./logs', 'tests', str(time()))
tensorboard = TensorBoard(log_dir=log_dir)
names=['a', 'b']
values=[0, 1]
write_log(tensorboard, names, values, 0)