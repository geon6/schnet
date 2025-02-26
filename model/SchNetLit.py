import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch_geometric
import pytorch_lightning as pl

from .SchNet import SchNet
from util.config import conf
from torchmetrics import (
    MeanSquaredError, 
    MeanAbsoluteError,
)


class SchNetLit(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SchNet(**conf.schnet)
        self.criterion = nn.MSELoss()
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()

        self.save_hyperparameters(conf)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=conf.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=conf.step_size, gamma=conf.gamma)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    def training_step(self, batch, batch_idx):
        pred = self.model(batch)
        label = batch.y[:, conf.target].view((-1, 1))
        loss = self.criterion(pred, label)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch)
        label = batch.y[:, conf.target].view((-1, 1))
        loss = self.criterion(pred, label)
        self.mae.update(pred, label)
        self.mse.update(pred, label)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        pred = self.model(batch)
        label = batch.y[:, conf.target].view((-1, 1))
        loss = self.criterion(pred, label)
        self.mae.update(pred, label)
        self.mse.update(pred, label)
        return {
            'loss': loss,
            'mae': self.mae.compute(),
            'mse': self.mse.compute(),
        }
        
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log('train_loss', outputs['loss'], 
                prog_bar=True, # 显示在进度条
                logger=True, # 记录到 logger
                batch_size=conf.batch_size
        )

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        self.log('val_loss', outputs['loss'], logger=True, on_epoch=True, batch_size=conf.batch_size)
    
    def on_validation_epoch_end(self):
        mae = self.mae.compute()
        mse = self.mse.compute()
        self.log('val_mae', mae, logger=True, batch_size=conf.batch_size)
        self.log('val_mse', mse, logger=True, batch_size=conf.batch_size)
        self.mae.reset()
        self.mse.reset()

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        self.log('test_loss', outputs['loss'], logger=True, on_epoch=True, batch_size=conf.batch_size)

    def on_test_epoch_end(self):
        mae = self.mae.compute()
        mse = self.mse.compute()
        self.log('test_mae', mae, logger=True, batch_size=conf.batch_size)
        self.log('test_mse', mse, logger=True, batch_size=conf.batch_size)
        self.mae.reset()
        self.mse.reset()

