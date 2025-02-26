import pytorch_lightning as pl

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from util.config import conf
from torch_geometric.transforms import (
    RadiusGraph,
    Compose,
)

class QM9DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset = QM9(
            root=conf.data_dir,
            transform=Compose([
                RadiusGraph(conf.cutoff),
            ])
        )

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self.dataset[:100000]
            self.val_dataset = self.dataset[100000:110000]
        if stage == 'test' or stage is None:
            self.test_dataset = self.dataset[110000:120000]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=conf.batch_size, num_workers=conf.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=conf.batch_size, num_workers=conf.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=conf.batch_size, num_workers=conf.num_workers)
