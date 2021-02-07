from aodnet.training import Trainer
from aodnet.utils import download_dataset, init_wandb


download_dataset()
init_wandb(
    project_name='aodnet', experiment_name='aodnet_train_nyu',
    wandb_api_key='cf0947ccde62903d4df0742a58b8a54ca4c11673'
)
trainer = Trainer()
trainer.build_datasets(
    dataset_path='/Users/soumikrakshit/Workspace/datasets/Dehazing',
    image_crop_size=256, buffer_size=1024, batch_size=4, val_split=0.1
)
trainer.build_model(build_shape=(1, 256, 256, 3))
trainer.train(epochs=10)
