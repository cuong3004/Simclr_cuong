import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import os

folder_checkpoint = "checkpoint" 
if not os.path.exists(folder_checkpoint): # create folder
    os.mkdir(folder_checkpoint)


# class InputMonitor(pl.Callback):

#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
#         # return
#         if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            
#             x, y = batch

#             logger = trainer.logger
#             logger.experiment.add_histogram("input", x, global_step=trainer.global_step)
#             logger.experiment.add_histogram("target", y, global_step=trainer.global_step)


checkpoint_callback = ModelCheckpoint(
    dirpath=folder_checkpoint,
    filename="model",
    save_top_k=1,
    verbose=False,
    monitor='val_los',
    mode='min',    
)

# early_stop_callback = EarlyStopping(monitor="valid_loss", patience=3, verbose=False, mode="min")

# input_monitor = InputMonitor()