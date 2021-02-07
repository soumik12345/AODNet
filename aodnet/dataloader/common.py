import os
from glob import glob
import tensorflow as tf
from typing import Tuple, List


def get_image_file_list(dataset_path: str) -> Tuple[List[str], List[str]]:
    original_image_files = []
    hazy_image_paths = sorted(glob(str(os.path.join(dataset_path, 'train_images/*.jpg'))))
    for image_path in hazy_image_paths:
        image_file_name = image_path.split('/')[-1]
        original_file_name = image_file_name.split('_')[0] + '_' + image_file_name.split('_')[1] + '.jpg'
        original_image_files.append(str(os.path.join(
            dataset_path, 'original_images/' + original_file_name)))
    return original_image_files, hazy_image_paths


def read_images(image_files):
    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    dataset = dataset.map(tf.io.read_file)
    dataset = dataset.map(
        lambda x: tf.image.decode_png(x, channels=3),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return dataset


def random_crop(low_image, enhanced_image, low_crop_size, enhanced_crop_size):
    low_image_shape = tf.shape(low_image)[:2]
    low_w = tf.random.uniform(
        shape=(), maxval=low_image_shape[1] - low_crop_size + 1, dtype=tf.int32)
    low_h = tf.random.uniform(
        shape=(), maxval=low_image_shape[0] - low_crop_size + 1, dtype=tf.int32)
    enhanced_w = low_w
    enhanced_h = low_h
    low_image_cropped = low_image[
                        low_h:low_h + low_crop_size,
                        low_w:low_w + low_crop_size
                        ]
    enhanced_image_cropped = enhanced_image[
                             enhanced_h:enhanced_h + enhanced_crop_size,
                             enhanced_w:enhanced_w + enhanced_crop_size
                             ]
    return low_image_cropped, enhanced_image_cropped


def random_flip(low_image, enhanced_image):
    return tf.cond(
        tf.random.uniform(shape=(), maxval=1) < 0.5,
        lambda: (low_image, enhanced_image),
        lambda: (
            tf.image.flip_left_right(low_image),
            tf.image.flip_left_right(enhanced_image)
        )
    )


def random_rotate(low_image, enhanced_image):
    condition = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(low_image, condition), tf.image.rot90(enhanced_image, condition)


def apply_scaling(low_image, enhanced_image):
    low_image = tf.cast(low_image, tf.float32)
    enhanced_image = tf.cast(enhanced_image, tf.float32)
    low_image = low_image / 255.0
    enhanced_image = enhanced_image / 255.0
    return low_image, enhanced_image


def configure_dataset(dataset, image_crop_size: int, buffer_size: int, batch_size: int, is_dataset_train: bool):
    dataset = dataset.map(
        apply_scaling, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if image_crop_size > 0:
        dataset = dataset.map(
            lambda low, high: random_crop(
                low, high, image_crop_size, image_crop_size
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    if is_dataset_train:
        dataset = dataset.map(
            random_rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
