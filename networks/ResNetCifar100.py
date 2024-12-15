import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import ResNetCifar10

class ResNetBottom(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNetBottom, self).__init__()
        self.model = ResNetCifar10.ResNetBottom(ResNetCifar10.ResBlock, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
class ResNetTop(nn.Module):
    def __init__(self, n_client, num_classes=100):
        super(ResNetTop, self).__init__()
        self.model = ResNetCifar10.ResNetTop(n_client, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x