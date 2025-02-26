import torch
import torch.nn as nn
import torch.nn.functional as F

from .CFConv import CFConv
from .ShiftedSoftplus import ShiftedSoftplus


class Interaction(nn.Module):
    def __init__(
        self, 
        hidden_dim = 64, 
        rbf_dim = 300
    ):
        super().__init__()
        self.atom_wise1 = nn.Linear(hidden_dim, hidden_dim)
        self.cfconv = CFConv(hidden_dim, rbf_dim)
        self.atom_wise2 = nn.Linear(hidden_dim, hidden_dim)
        self.ssf = ShiftedSoftplus()
        self.atom_wise3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, batch):
        residual = x
        x = self.atom_wise1(x)
        x = self.cfconv(x, batch)
        x = self.atom_wise2(x)
        x = self.ssf(x)
        x = self.atom_wise3(x)
        return x + residual
