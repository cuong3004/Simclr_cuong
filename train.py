import numpy as np
import pytorch_lightning as pl
from lit_module import SimCLR
from data_module import CelebADataModule
from utils import SimCLREvalDataTransform, SimCLRTrainDataTransform, test_transform
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms
import torch

batch_size = 512
input_height = 112

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

test_transform(dm.train_dataloader)

# def test_transform(dataset, wandb_logger):
    

# print(len(next(iter(dm.train_dataloader()))))

# gpus = 1 if torch.cuda.is_available() else 0

# model = SimCLR(gpus=gpus, dataset="", num_samples=dm.num_samples, batch_size=dm.batch_size)


# # from pytorch_lightning import Trainer

# wandb_logger = WandbLogger(name="Face", log_model="all", project="Simclr")
# # trainer = Trainer(logger=wandb_logger)

# trainer = pl.Trainer(gpus=gpus, logger=wandb_logger)
# trainer.fit(model, dm)