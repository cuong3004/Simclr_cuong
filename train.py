import numpy as np
import pytorch_lightning as pl
from lit_module import SimCLR
from data_module import CelebADataModule
from utils import SimCLREvalDataTransform, SimCLRTrainDataTransform, test_transform
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms
from callbacks import checkpoint_callback
import torch

batch_size = 1536
input_height = 112
max_epoch = 50

data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(input_height),
    transforms.RandAugment(),
])

final_transforms = transforms.Compose([
        transforms.ToTensor(),          
        transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

sim_transforms = transforms.Compose([
        data_transforms,
        final_transforms
])
# val_transforms = transforms.Compose
sim_train_transforms = SimCLRTrainDataTransform(batch_size)
sim_val_transforms = SimCLREvalDataTransform(batch_size)

sim_train_transforms.train_transform = sim_transforms
sim_val_transforms.train_transform = sim_transforms



dm = CelebADataModule("data/CelebA/img_align_celeba", num_workers=2)
dm.train_transforms = sim_train_transforms
dm.val_transforms = sim_val_transforms

test_transform(dm.train_dataloader())

# import wandb
# import os
# run = wandb.init()
# artifact = run.use_artifact('duongcuong1977/Simclr/model-s00ad5y4:v4', type='model')
# artifact_dir = artifact.download()

# model = SimCLR.load_from_checkpoint(os.path.join("./artifacts/model-s00ad5y4:v4", "model.ckpt"))

# gpus = 1 if torch.cuda.is_available() else 0
# wandb_logger = WandbLogger(name="Face_randaugument_batch_512_t_0.1", log_model="all", project="Simclr", id="s00ad5y4")
# trainer = pl.Trainer(gpus=gpus, logger=wandb_logger, callbacks=[checkpoint_callback], resume_from_checkpoint=os.path.join("./artifacts/model-s00ad5y4:v4", "model.ckpt"), max_epochs=max_epoch)
# trainer.fit(model, dm)
# trainer.validate(model, dm)
# def test_transform(dataset, wandb_logger):
    

# print(len(next(iter(dm.train_dataloader()))))

gpus = 1 if torch.cuda.is_available() else 0

model = SimCLR(gpus=gpus, dataset="", num_samples=dm.num_samples, batch_size=dm.batch_size)


# from pytorch_lightning import Trainer

wandb_logger = WandbLogger(name=f"Face_randaugument_batch_{batch_size}", log_model="all", project="Simclr")
# trainer = Trainer(logger=wandb_logger)

trainer = pl.Trainer(gpus=gpus, logger=wandb_logger)
trainer.fit(model, dm)