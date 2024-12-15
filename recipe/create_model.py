import torch
from networks.ResNetCifar10 import ResBlock, ResNetBottom, ResNetTop
from parties.Client import Client
from networks import ResNetCifar100
from parties.Host import Host

class Create_model:
    def __init__(self, num_clients, device):
        self.num = num_clients
        self.device = device

    def create_modelForCIFAR10(self):
        clients = []
        Model = ResNetTop(self.num)
        host = Host(Model)
        host = host.to(self.device)
        for i in range(self.num):
            client = Client(ResNetBottom(ResBlock))
            client = client.to(self.device)
            clients.append(client)
        return clients, host
    
    def create_modelForCIFAR100(self):
        clients = []
        Model = ResNetCifar100.ResNetTop(self.num)
        host = Host(Model)
        host = host.to(self.device)
        for i in range(self.num):
            client = Client(ResNetCifar100.ResNetBottom())
            client = client.to(self.device)
            clients.append(client)
        return clients, host
    