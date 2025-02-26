import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        shift = torch.log(torch.tensor(2.0))
        self.register_buffer('shift', shift)

    def forward(self, x: Tensor) -> Tensor:
        # ln(0.5 + 0.5 * exp(x)) = ln(1 + exp(x)) - ln(2) = softplus(x) - ln(2)
        return self.softplus(x) - self.shift
