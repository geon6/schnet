import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch
from .ShiftedSoftplus import ShiftedSoftplus


class CFConv(MessagePassing):
    def __init__(self, hidden_dim = 64, rbf_dim = 300):
        super().__init__()
        self.rbf = RBF(rbf_dim)
        self.dense1 = nn.Linear(rbf_dim, hidden_dim)
        self.ssf1 = ShiftedSoftplus()
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.ssf2 = ShiftedSoftplus()

    def forward(self, x: Tensor, batch: Batch) -> Tensor:
        row, col = batch.edge_index
        pos: Tensor = batch.pos
        dist = (pos[row] - pos[col]).norm(dim=-1)
        p = self.rbf(dist)
        p = self.dense1(p)
        p = self.ssf1(p)
        p = self.dense2(p)
        p = self.ssf2(p)
        W = p * dist.view(-1, 1)
        x = x + self.propagate(edge_index=batch.edge_index, x=x, W=W)
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class RBF(nn.Module):
    def __init__(self, rbf_dim = 300, gamma = 10.0):
        super().__init__()
        gamma = torch.tensor(gamma)
        self.register_buffer('gamma', gamma)
        mu_k = torch.linspace(0.0, 30.0, rbf_dim)
        self.register_buffer('mu_k', mu_k)

    def forward(self, dist: Tensor) -> Tensor:
        # e_k(r_i - r_j) = exp(-gamma * (d_{ij} - mu_k)^2)
        dist = dist.view(-1, 1)
        mu_k = self.mu_k.view(1, -1)
        return torch.exp(-self.gamma * torch.pow(dist - mu_k, 2))
