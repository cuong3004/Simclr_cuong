import numpy as np
import pytorch_lightning as pl
from lit_module import SimCLR
from data_module import CelebADataModule
from utils import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pytorch_lightning.loggers import WandbLogger
import torch

batch_size = 256
dm = CelebADataModule("data/CelebA/img_align_celeba", num_workers=2)
dm.train_transforms = SimCLRTrainDataTransform(batch_size)
dm.val_transforms = SimCLREvalDataTransform(batch_size)

# print(len(next(iter(dm.train_dataloader()))))

gpus = 1 if torch.cuda.is_available() else 0

model = SimCLR(gpus=gpus, dataset="", num_samples=dm.num_samples, batch_size=dm.batch_size)


# from pytorch_lightning import Trainer

wandb_logger = WandbLogger(name="Face", log_model="all", project="Simclr")
# trainer = Trainer(logger=wandb_logger)

trainer = pl.Trainer(gpus=gpus, logger=wandb_logger)
trainer.fit(model, dm)