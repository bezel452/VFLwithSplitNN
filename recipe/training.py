import torch
from recipe.create_model import Create_model
import torch.nn as nn
import torch.optim as optim
from data.data_preprocessing import split_featurForCifar10, loaderCifar10, loaderCifar100, split_featurForCifar100, split_featurForCinic10, loaderCinic10, loaderImageNette, split_featurForImageNette
import csv
import time

class Training:
    def __init__(self, num_clients, file_path, epochs, batch_size, lr, momentum, device):
        self.num = num_clients
        self.device = device
        self.file = file_path
        self.eps = epochs
        self.bs = batch_size
        self.lr = lr
        self.mo = momentum

    def bottom_loss(self, y1, y2):
        return torch.sum(y1 * y2)
    
    def trainingCIFAR10(self):
        create = Create_model(self.num, self.device)
        clients, host = create.create_modelForCIFAR10()
        Top_loss = nn.CrossEntropyLoss()
        trainloader, testloader = loaderCifar10(self.file, self.bs)
        self.training(clients=clients, host=host, Top_loss=Top_loss, trainloader=trainloader, testloader=testloader, mode="CIFAR10", lr=self.lr, momentum=self.mo)
        
    def trainingCIFAR100(self):
        create = Create_model(self.num, self.device)
        clients, host = create.create_modelForCIFAR100()
        Top_loss = nn.CrossEntropyLoss()
        trainloader, testloader = loaderCifar100(self.file, self.bs)
        self.training(clients=clients, host=host, Top_loss=Top_loss, trainloader=trainloader, testloader=testloader, mode="CIFAR100", lr=self.lr, momentum=self.mo)

    def trainingCINIC10(self):
        create = Create_model(self.num, self.device)
    #    clients, host = create.create_modelForCINIC10()
        clients, host = create.create_modelForCINIC10p()
        Top_loss = nn.CrossEntropyLoss()
        trainloader, testloader = loaderCinic10(self.file, self.bs)
        self.training(clients=clients, host=host, Top_loss=Top_loss, trainloader=trainloader, testloader=testloader, mode="CINIC10", lr=self.lr, momentum=self.mo)

    def trainingImageNette(self):
        create = Create_model(self.num, self.device)
        clients, host = create.create_modelForImageNette()
        Top_loss = nn.CrossEntropyLoss()
        trainloader, testloader = loaderImageNette(self.file, self.bs)
        self.training(clients=clients, host=host, Top_loss=Top_loss, trainloader=trainloader, testloader=testloader, mode="ImageNette", lr=self.lr, momentum=self.mo)

    def training(self, clients, host, Top_loss, trainloader, testloader, mode, lr, momentum):
        clients_optim = []
        losses = []
        for client in clients:
            optimizer = optim.SGD(params=client.parameters(), lr=lr, momentum=momentum)
            clients_optim.append(optimizer)
        host_optim = optim.SGD(params=host.parameters(), lr=lr, momentum=momentum)
        
        for epoch in range(self.eps):
            print("----------Training----------")
            loss_per_epoch = 0.0
            for i, (feature, label) in enumerate(trainloader):
                feature, label = feature.to(self.device), label.to(self.device)
                if mode == 'CIFAR10':
                    x = split_featurForCifar10(feature, self.num)
                elif mode == 'CIFAR100':
                    x = split_featurForCifar100(feature, self.num)
                elif mode == 'CINIC10':
                    x = split_featurForCinic10(feature, self.num)
                elif mode == 'ImageNette':
                    x = split_featurForImageNette(feature, self.num)
                else:
                    raise Exception("Error! Unknown Dataset.")
                inputs = []
                outputs = []
                for j in range(self.num):
                    inputPer = torch.tensor([], requires_grad=True)
                    
                    clients[j].train()
                    clients_optim[j].zero_grad()
                    out = clients[j](x[j])
                    inputPer.data = out.data
                    outputs.append(out)
                    inputs.append(inputPer)
                host.train()
                host_optim.zero_grad()
                host_out = host(inputs)
                host_loss = Top_loss(host_out, label)
                host_loss.backward()
                host_optim.step()
                for j in range(self.num):
                    loss = self.bottom_loss(outputs[j], inputs[j].grad)
                    loss.backward()
                    clients_optim[j].step()
                loss_per_epoch += host_loss
                _, predicted = torch.max(host_out.data, 1)
                print(f"Train Epoch:[{epoch + 1} | {self.eps}] | Loss: {host_loss.data.item()} | Acc: {(predicted == label).sum() / len(predicted)}")
            self.validating(testloader, clients, host, mode)
            loss_per_epoch = loss_per_epoch / len(trainloader)
            losses.append(loss_per_epoch.cpu().detach().numpy())
        self.save_csv(losses, mode)

    def validating(self, testloader, clients, host, mode):
        with torch.no_grad():
            total = 0
            correct = 0
            for i, (feature, label) in enumerate(testloader):
                feature, label = feature.to(self.device), label.to(self.device)
                if mode == 'CIFAR10':
                    x = split_featurForCifar10(feature, self.num)
                elif mode == 'CIFAR100':
                    x = split_featurForCifar100(feature, self.num)
                elif mode == 'CINIC10':
                    x = split_featurForCinic10(feature, self.num)
                elif mode == 'ImageNette':
                    x = split_featurForImageNette(feature, self.num)
                else:
                    raise Exception("Error! Unknown Dataset.")
                inputs = []
                for j in range(self.num):
                    clients[j].eval()
                    inputs.append(clients[j](x[j]))
                host.eval()
                output = host(inputs)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum()
            print("----------Validating----------")
            print(f'Accuracy on the TEST set is: {(100 * correct / total)}%') 

    def save_csv(self, losses, dataset):
        t = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        with open(dataset + t + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(losses)




        
    

