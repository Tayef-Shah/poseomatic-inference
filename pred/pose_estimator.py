import requests
import tensorflow as tf
from PIL import Image
from pred.models.movenet import movenet


def load_image(image_url):
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
        return image
    except Exception as e:
        print(e)
        print("image could not be opened")


def preprocess_img(image):
    image = tf.keras.utils.img_to_array(image)
    image = tf.convert_to_tensor(image)

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    input_size = 192
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    return input_image


def crop_region(img):
    pass


def estimate_pose(image):
    # will later become loop for video
    keypoints = movenet(image)
    return keypoints
