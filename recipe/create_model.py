import torch
from networks.ResNetCifar10 import ResBlock, ResNetBottom, ResNetTop
from parties.Client import Client
from networks import DenseNetImagenette, ResNetCINIC10, ResNetCifar100, LRBHI, ResNetCINIC10p
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
    
    def create_modelForCINIC10(self):
        clients = []
        Model = ResNetCINIC10.ResNetTop(self.num)
        host = Host(Model)
        host = host.to(self.device)
        for i in range(self.num):
            client = Client(ResNetCINIC10.ResNetBottom())
            client = client.to(self.device)
            clients.append(client)
        return clients, host
    def create_modelForCINIC10p(self):
        clients = []
        Model = ResNetCINIC10p.ResNetTop(self.num)
        host = Host(Model)
        host = host.to(self.device)
        for i in range(self.num):
            client = Client(ResNetCINIC10p.ResNetBottom())
            client = client.to(self.device)
            clients.append(client)
        return clients, host
    
    def create_modelForImageNette(self):
        clients = []
        Model = DenseNetImagenette.DenseNetTop(self.num)
        host = Host(Model)
        host = host.to(self.device)
        for i in range(self.num):
            client = Client(DenseNetImagenette.DenseNetBottom())
            client = client.to(self.device)
            clients.append(client)
        return clients, host
    
    def create_modelForBHI(self):
        clients = []
        Model = LRBHI.LRTop(self.num)
        host = Host(Model)
        host = host.to(self.device)
        for i in range(self.num):
            client = Client(LRBHI.LogisticRegressionBottom(30 // self.num, 2))
            client = client.to(self.device)
            clients.append(client)
        return clients, host