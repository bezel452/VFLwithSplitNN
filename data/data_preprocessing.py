import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch

def split_featurForCifar10(feature, num_clients):
    x = []
    lenPerParty = 32 // num_clients
    nowLen = 0
    for i in range(num_clients):
        if i == num_clients - 1:
            x.append(feature[:, :, :, nowLen:32])
        else:
            x.append(feature[:, :, :, nowLen:nowLen+lenPerParty])
            nowLen += lenPerParty
    return x

def split_featurForCifar100(feature, num_clients):
    x = []
    lenPerParty = 32 // num_clients
    nowLen = 0
    for i in range(num_clients):
        if i == num_clients - 1:
            x.append(feature[:, :, :, nowLen:32])
        else:
            x.append(feature[:, :, :, nowLen:nowLen+lenPerParty])
            nowLen += lenPerParty
    return x

def loaderCifar10(file_path, batch_size):
    transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root=file_path, train=True, download=False, transform=transforms_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=file_path, train=False, download=False, transform=transforms_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

def loaderCifar100(file_path, batch_size):
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transforms_test = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    trainset = torchvision.datasets.CIFAR100(root=file_path, train=True, download=True, transform=transforms_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root=file_path, train=False, download=True, transform=transforms_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader
