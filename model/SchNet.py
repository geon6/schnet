import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.resolver import aggregation_resolver
from .Interaction import Interaction
from .ShiftedSoftplus import ShiftedSoftplus


class SchNet(nn.Module):
    def __init__(
        self,
        hidden_dim = 64,
        rbf_dim = 300,
        num_interactions = 3,
        interaction_dim = 64,
        atom_dim = 32,
        readout = 'add'
    ):
        super().__init__()
        self.emb = nn.Embedding(100, hidden_dim)
        self.interactions = nn.ModuleList([
            Interaction(hidden_dim, rbf_dim) for _ in range(num_interactions)
        ])
        self.atom_wise1 = nn.Linear(interaction_dim, atom_dim)
        self.ssp = ShiftedSoftplus()
        self.atom_wise2 = nn.Linear(atom_dim, 1)
        self.readout = aggregation_resolver(readout)

    def forward(self, batch):
        x = self.emb(batch.z)
        for interaction in self.interactions:
            x = interaction(x, batch)
        x = self.atom_wise1(x)
        x = self.ssp(x)
        x = self.atom_wise2(x)
        return self.readout(x, batch.batch)
