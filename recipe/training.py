import torch
from recipe.create_model import Create_model
import torch.nn as nn
import torch.optim as optim
from data.data_preprocessing import split_featurForCifar10, loaderCifar10, loaderCifar100, split_featurForCifar100, split_featurForCinic10, loaderCinic10, loaderImageNeet, split_featurForImageNette
import csv
import time

class Training:
    def __init__(self, num_clients, file_path, epochs, batch_size, device):
        self.num = num_clients
        self.device = device
        self.file = file_path
        self.eps = epochs
        self.bs = batch_size

    def bottom_loss(self, y1, y2):
        return torch.sum(y1 * y2)
    
    def trainingCIFAR10(self):
        create = Create_model(self.num, self.device)
        clients, host = create.create_modelForCIFAR10()
        Top_loss = nn.CrossEntropyLoss()
        clients_optim = []
        losses = []
        for client in clients:
            optimizer = optim.SGD(params=client.parameters(), lr=0.01, momentum=0.9)
            clients_optim.append(optimizer)
        host_optim = optim.SGD(params=host.parameters(), lr=0.01, momentum=0.9)
        trainloader, testloader = loaderCifar10(self.file, self.bs)
        for epoch in range(self.eps):
            print("----------Training----------")
            loss_per_epoch = 0.0
            for i, (feature, label) in enumerate(trainloader):
                feature, label = feature.to(self.device), label.to(self.device)
                x = split_featurForCifar10(feature, self.num)
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
                print(f"Train Epoch:[{epoch + 1} | {self.eps}] | Loss: {host_loss.data.item()}")
            self.validatingCIFAR10(testloader, clients, host)
            loss_per_epoch = loss_per_epoch / len(trainloader)
            losses.append(loss_per_epoch.cpu().detach().numpy())
        self.save_csv(losses, 'CIFAR10')

    def validatingCIFAR10(self, testloader, clients, host):
        with torch.no_grad():
            total = 0
            correct = 0
            for i, (feature, label) in enumerate(testloader):
                feature, label = feature.to(self.device), label.to(self.device)
                x = split_featurForCifar10(feature, self.num)
                inputs = []
                for j in range(self.num):
                    inputs.append(clients[j](x[j]))
                output = host(inputs)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum()
            print("----------Validating----------")
            print(f'Accuracy on the TEST set is: {(100 * correct / total)}%')
    
    def trainingCIFAR100(self):
        create = Create_model(self.num, self.device)
        clients, host = create.create_modelForCIFAR100()
        Top_loss = nn.CrossEntropyLoss()
        clients_optim = []
        losses = []
        for client in clients:
            optimizer = optim.SGD(params=client.parameters(), lr=0.01, momentum=0.9)
            clients_optim.append(optimizer)
        host_optim = optim.SGD(params=host.parameters(), lr=0.01, momentum=0.9)
        trainloader, testloader = loaderCifar100(self.file, self.bs)
        for epoch in range(self.eps):
            print("----------Training----------")
            loss_per_epoch = 0.0
            for i, (feature, label) in enumerate(trainloader):
                feature, label = feature.to(self.device), label.to(self.device)
                x = split_featurForCifar100(feature, self.num)
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
                print(f"Train Epoch:[{epoch + 1} | {self.eps}] | Loss: {host_loss.data.item()}")
            self.validatingCIFAR100(testloader, clients, host)
            loss_per_epoch = loss_per_epoch / len(trainloader)
            losses.append(loss_per_epoch.cpu().detach().numpy())
        self.save_csv(losses, 'CIFAR100')

    def validatingCIFAR100(self, testloader, clients, host):
        with torch.no_grad():
            total = 0
            correct = 0
            for i, (feature, label) in enumerate(testloader):
                feature, label = feature.to(self.device), label.to(self.device)
                x = split_featurForCifar100(feature, self.num)
                inputs = []
                for j in range(self.num):
                    inputs.append(clients[j](x[j]))
                output = host(inputs)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum()
            print("----------Validating----------")
            print(f'Accuracy on the TEST set is: {(100 * correct / total)}%')

    def trainingCINIC10(self):
        create = Create_model(self.num, self.device)
        clients, host = create.create_modelForCINIC10()
        Top_loss = nn.CrossEntropyLoss()
        clients_optim = []
        losses = []
        for client in clients:
            optimizer = optim.SGD(params=client.parameters(), lr=0.01, momentum=0.9)
            clients_optim.append(optimizer)
        host_optim = optim.SGD(params=host.parameters(), lr=0.01, momentum=0.9)
        trainloader, testloader = loaderCinic10(self.file, self.bs)
        for epoch in range(self.eps):
            print("----------Training----------")
            loss_per_epoch = 0.0
            for i, (feature, label) in enumerate(trainloader):
                feature, label = feature.to(self.device), label.to(self.device)
                x = split_featurForCinic10(feature, self.num)
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
                print(f"Train Epoch:[{epoch + 1} | {self.eps}] | Loss: {host_loss.data.item()}")
            self.validatingCINIC10(testloader, clients, host)
            loss_per_epoch = loss_per_epoch / len(trainloader)
            losses.append(loss_per_epoch.cpu().detach().numpy())
        self.save_csv(losses, 'CINIC10')

    def validatingCINIC10(self, testloader, clients, host):
        with torch.no_grad():
            total = 0
            correct = 0
            for i, (feature, label) in enumerate(testloader):
                feature, label = feature.to(self.device), label.to(self.device)
                x = split_featurForCinic10(feature, self.num)
                inputs = []
                for j in range(self.num):
                    inputs.append(clients[j](x[j]))
                output = host(inputs)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum()
            print("----------Validating----------")
            print(f'Accuracy on the TEST set is: {(100 * correct / total)}%')

    def trainingImageNette(self):
        create = Create_model(self.num, self.device)
        clients, host = create.create_modelForImageNette()
        Top_loss = nn.CrossEntropyLoss()
        clients_optim = []
        losses = []
        for client in clients:
            optimizer = optim.SGD(params=client.parameters(), lr=0.01, momentum=0.9)
            clients_optim.append(optimizer)
        host_optim = optim.SGD(params=host.parameters(), lr=0.01, momentum=0.9)
        trainloader, testloader = loaderImageNeet(self.file, self.bs)
        for epoch in range(self.eps):
            print("----------Training----------")
            loss_per_epoch = 0.0
            for i, (feature, label) in enumerate(trainloader):
                feature, label = feature.to(self.device), label.to(self.device)
                x = split_featurForImageNette(feature, self.num)
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
                print(f"Train Epoch:[{epoch + 1} | {self.eps}] | Loss: {host_loss.data.item()}")
            self.validatingImageNette(testloader, clients, host)
            loss_per_epoch = loss_per_epoch / len(trainloader)
            losses.append(loss_per_epoch.cpu().detach().numpy())
        self.save_csv(losses, 'ImageNette')

    def validatingImageNette(self, testloader, clients, host):
        with torch.no_grad():
            total = 0
            correct = 0
            for i, (feature, label) in enumerate(testloader):
                feature, label = feature.to(self.device), label.to(self.device)
                x = split_featurForImageNette(feature, self.num)
                inputs = []
                for j in range(self.num):
                    inputs.append(clients[j](x[j]))
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




        
    

