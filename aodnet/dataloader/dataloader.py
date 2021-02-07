from .common import *


class DeHazeDataLoader:

    def __init__(self, dataset_path: str):
        self.original_images, self.hazy_images = get_image_file_list(dataset_path=dataset_path)

    def __len__(self):
        assert len(self.hazy_images) == len(self.original_images)
        return len(self.hazy_images)

    def build_dataset(self, image_crop_size: int, buffer_size: int, batch_size: int, val_split: float):
        hazy_dataset = read_images(self.hazy_images)
        original_dataset = read_images(self.original_images)
        dataset = tf.data.Dataset.zip((hazy_dataset, original_dataset))
        cardinality = tf.data.experimental.cardinality(dataset).numpy()
        train_dataset = configure_dataset(
            dataset=dataset.skip(int(cardinality * val_split)),
            image_crop_size=image_crop_size, buffer_size=buffer_size,
            batch_size=batch_size, is_dataset_train=True
        )
        val_dataset = configure_dataset(
            dataset=dataset.take(int(cardinality * val_split)),
            image_crop_size=image_crop_size, buffer_size=buffer_size,
            batch_size=batch_size, is_dataset_train=True
        )
        return train_dataset, val_dataset
