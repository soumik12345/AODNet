import numpy as np
from PIL import Image
import tensorflow as tf

from .models import AODNet
from .utils import download_from_drive


class Inferer:

    def __init__(self):
        self.model = None

    def build_model(self):
        self.model = AODNet()
        download_from_drive(
            file_id='1-B1RuIrhCBY9T5XALQFS8ZHX-Scq7CEV',
            file_name='aodnet_weights.zip', unpack_location='./'
        )
        self.model.load_weights(
            './checkpoints/aodnet_weights/weights')

    def infer(self, image_path):
        original_image = Image.open(image_path)
        image = tf.keras.preprocessing.image.img_to_array(original_image)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = self.model.predict(image)
        return original_image, prediction[0]
