import torch
import torch.nn as nn
from parties.Party import Party

class Client(Party):
    def __init__(self, Model):
        super(Client, self).__init__(Model)
        self.model = Model

    def forward(self, x):
        return self.model(x)