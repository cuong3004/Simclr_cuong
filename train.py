import numpy as np
import pytorch_lightning as pl
from lit_module import SimCLR
from data_module import CelebADataModule
from utils import SimCLREvalDataTransform, SimCLRTrainDataTransform, test_transform
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms
from callbacks import checkpoint_callback
import torch
import os
import wandb

batch_size = 256
input_height = 112
max_epoch = 50

main_trans = transforms.AutoAugment()
name_trans = str(main_trans).split("(")[0]

project = "SimCLR"
name = f"{name_trans}_{batch_size}"



data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(input_height),
    main_trans,
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

gpus = 1 if torch.cuda.is_available() else 0

wandb_logger = WandbLogger(log_model="all")

if True:
    
    run = wandb.init(name=name, project=project, id=id) 
    model = SimCLR(gpus=gpus, dataset="", num_samples=dm.num_samples, batch_size=dm.batch_size)
    trainer = pl.Trainer(gpus=gpus, logger=wandb_logger)
    trainer.fit(model, dm)

else:

    model = SimCLR.load_from_checkpoint(path_checkpoint)
    trainer = pl.Trainer(gpus=gpus, logger=wandb_logger, callbacks=[checkpoint_callback], resume_from_checkpoint=path_checkpoint)
    trainer.fit(model, dm)
    trainer.validate(model, dm)


