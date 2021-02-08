import tensorflow as tf
from typing import Tuple
from .models import AODNet
from datetime import datetime
from wandb.keras import WandbCallback
from plotly import graph_objects as go
from .dataloader import DeHazeDataLoader
from .utils import peak_signal_noise_ratio, plot_result


class Trainer:

    def __init__(self):
        self.train_dataset = None
        self.val_dataset = None
        self.model = None
        self.training_history = None

    def __len__(self):
        return len(self.train_dataset)

    def _plot_dataset_samples(self):
        print(self.train_dataset)
        print(self.val_dataset)
        x, y = next(iter(self.train_dataset))
        print('X shape:', x.shape)
        print('Y shape:', y.shape)
        num = x.shape[0] if x.shape[0] <= 4 else 4
        for _ in range(num):
            plot_result(x.numpy()[_], y.numpy()[_], 'Hazy', 'Original')

    def _plot_history(self, train_property: str, plot_title: str):
        figure = go.Figure()
        figure.add_traces(
            go.Scatter(
                x=[i + 1 for i in list(
                    range(len(self.training_history.history[train_property])))],
                y=self.training_history.history[train_property],
                mode='lines+markers', name='Training Result: ' + train_property
            )
        )
        figure.update_layout(title='Loss')
        return figure

    def build_datasets(
            self, dataset_path: str, image_crop_size: int, buffer_size: int,
            batch_size: int, val_split: float, plot_samples: bool):
        dataloader = DeHazeDataLoader(dataset_path=dataset_path)
        self.train_dataset, self.val_dataset = dataloader.build_dataset(
            image_crop_size=image_crop_size, buffer_size=buffer_size,
            batch_size=batch_size, val_split=val_split
        )
        if plot_samples:
            self._plot_dataset_samples()

    def build_model(
            self, build_shape: Tuple[int, int, int, int], stddev: float = 0.02,
            weight_decay: float = 1e-4, learning_rate: float = 1e-4):
        self.model = AODNet(stddev=stddev, weight_decay=weight_decay)
        self.model.build(build_shape)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.MeanSquaredError(), metrics=[peak_signal_noise_ratio]
        )
        self.model.summary()

    def train(self, checkpoint_dir: str = './checkpoints/', epochs: int = 10):
        log_dir = "./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1,
                update_freq=50, write_images=True
            ),
            WandbCallback()
        ]
        self.training_history = self.model.fit(
            self.train_dataset, validation_data=self.val_dataset,
            epochs=epochs, callbacks=callbacks
        )
        self._plot_history(
            train_property='loss', plot_title='Loss').show()
        self._plot_history(
            train_property='peak_signal_noise_ratio', plot_title='PSNR').show()

    def save_model(self, model_name: str):
        save_path = './checkpoints/{}'.format(model_name)
        print('Saving model at {}...'.format(save_path))
        self.model.save(save_path, save_format='tf')
        print('Done!!!')

    def save_weights(self, model_name: str):
        save_path = './checkpoints/{}'.format(model_name)
        print('Saving model weights at {}...'.format(save_path))
        self.model.save_weights(save_path, save_format='tf')
        print('Done!!!')
