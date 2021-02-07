from aodnet.training import Trainer


trainer = Trainer()
trainer.build_datasets(
    dataset_path='/Users/soumikrakshit/Workspace/datasets/Dehazing',
    image_crop_size=256, buffer_size=1024, batch_size=4, val_split=0.1
)
trainer.build_model(build_shape=(1, 256, 256, 3))
