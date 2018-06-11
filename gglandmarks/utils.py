from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

def load_image(image_path, target_size):
    if image_path != '':
        image = load_img(image_path, target_size=target_size)
        image_array = img_to_array(image)
        return image_array

    return None

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()