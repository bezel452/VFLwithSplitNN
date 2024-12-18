import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch
from data import dataSetup
import os
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

def split_featurForCinic10(feature, num_clients):
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

def img_format_2_rgb(x):
    return x.convert("RGB")

def loaderCinic10(file_path, batch_size):
    train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),  
        transforms.ToTensor(),        
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  
    ])
    trainset = torchvision.datasets.ImageFolder(os.path.join(file_path, 'train'), transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.ImageFolder(os.path.join(file_path, 'test'), transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


def split_featurForImageNette(feature, num_clients):
    x = []
    lenPerParty = 256 // num_clients
    nowLen = 0
    for i in range(num_clients):
        if i == num_clients - 1:
            x.append(feature[:, :, :, nowLen:256])
        else:
            x.append(feature[:, :, :, nowLen:nowLen+lenPerParty])
            nowLen += lenPerParty
    return x

def loaderImageNette(file_path, batch_size):
    transform = transforms.Compose([
        transforms.Resize(256),  # 调整大小为256x256像素
        transforms.CenterCrop(224),  # 中心裁剪为224x224像素
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])

    trainset = torchvision.datasets.ImageFolder(os.path.join(file_path, 'train'), transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.ImageFolder(os.path.join(file_path, 'val'), transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

def split_featureBHI(feature, num_clients):
    x = []
    lenPerParty = 30 // num_clients
    nowLen = 0
    for i in range(num_clients):
        if i == num_clients - 1:
            x.append(feature[:, nowLen:30].float())
        else:
            x.append(feature[:, nowLen:nowLen+lenPerParty].float())
            nowLen += lenPerParty
    return x

def loaderBHI(file_path, batch_size):
    breast = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(breast.data, breast.target, test_size=0.2, random_state=1)
    std = StandardScaler()
    X_train = std.fit_transform(X_train) # 标准化数据
    X_test = std.transform(X_test)
    trainset = dataSetup.BreastData(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = dataSetup.BreastData(X_test, y_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader