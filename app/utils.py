from keras.preprocessing.image import load_img, img_to_array

def load_image(image_path, target_size):
    if image_path != '':
        image = load_img(image_path, target_size=target_size)
        image_array = img_to_array(image)
        return image_array

    return None