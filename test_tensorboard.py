import tensorflow as tf
from gglandmarks.utils import TensorboardLogger
import os
import time

log_dir = os.path.join('./logs', 'tests')
logger = TensorboardLogger(log_dir)

for i in range(100_000):
    logger.log_scalar('my value', i/10, i)

    time.sleep(0.1)