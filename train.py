import comet_ml
import torch 
import pytorch_lightning as pl

from util.config import conf
from util.logger import get_pl_loggers
from util.callback import get_pl_callbacks
from data.qm9 import QM9DataModule
from model.SchNetLit import SchNetLit

if __name__ == '__main__':
    pl.seed_everything(conf.seed)

    dm = QM9DataModule()

    model = SchNetLit()

    trainer = pl.Trainer(
        max_epochs=conf.num_epochs,
        accelerator="auto",
        devices=conf.devices,
        logger=get_pl_loggers(),
        callbacks=get_pl_callbacks(),
        # enable_progress_bar=False,
    )

    trainer.fit(model, dm)

    trainer.test(model, dm)
