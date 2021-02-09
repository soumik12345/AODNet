import os
import wandb
import gdown
import zipfile
import subprocess
import tensorflow as tf
from matplotlib import pyplot as plt


def unzip(zip_file: str, extract_location: str):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_location)


def download_from_drive(file_id: str, file_name: str, unpack_location: str = './'):
    gdown.download(
        'https://drive.google.com/uc?id=' + file_id,
        file_name, quiet=False
    )
    print('Unpacking...')
    unzip(file_name, extract_location=unpack_location)
    subprocess.run(['rm', file_name])
    print('Done!!!')


def download_dataset():
    download_from_drive(
        file_id='1sInD9Ydq8-x7WwqehE0EyRknMdSFPOat',
        file_name='Dehaze-NYU.zip', unpack_location='./'
    )


def init_wandb(project_name, experiment_name, wandb_api_key):
    if project_name is not None and experiment_name is not None:
        os.environ['WANDB_API_KEY'] = wandb_api_key
        wandb.init(project=project_name, name=experiment_name, sync_tensorboard=True)


def peak_signal_noise_ratio(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)


def plot_result(image1, image2, title1, title2):
    fig = plt.figure(figsize=(12, 12))
    fig.add_subplot(1, 2, 1).set_title(title1)
    _ = plt.imshow(image1)
    fig.add_subplot(1, 2, 2).set_title(title2)
    _ = plt.imshow(image2)
    plt.show()
