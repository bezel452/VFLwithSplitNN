import torch
import torch.nn as nn

class Party(nn.Module):
    def __init__(self, Model):
        super(Party, self).__init__()
        self.model = Model

    def forward(self, x):
        return self.model(x)