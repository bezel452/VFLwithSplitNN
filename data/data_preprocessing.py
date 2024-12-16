import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch
from data import dataSetup

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
    normalize_cinic = transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                               std=[0.24205776, 0.23828046, 0.25874835])
    transforms_ = transforms.Compose([
        transforms.Lambda(img_format_2_rgb),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize_cinic
    ])
    trainset = dataSetup.CINIC10(root=file_path, split='train', transform=transforms_)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = dataSetup.CINIC10(root=file_path, split='test', transform=transforms_)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


def split_featurForImageNette(feature, num_clients):
    x = []
    lenPerParty = 224 // num_clients
    nowLen = 0
    for i in range(num_clients):
        if i == num_clients - 1:
            x.append(feature[:, :, :, nowLen:224])
        else:
            x.append(feature[:, :, :, nowLen:nowLen+lenPerParty])
            nowLen += lenPerParty
    return x

def loaderImageNeet(file_path, batch_size):
    preprocess = transforms.Compose([
        transforms.Lambda(img_format_2_rgb),
        transforms.Resize((224, 224)),  # 调整图像大小为224x224
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像数据
    ])
    trainset = dataSetup.ImageNette(root=file_path, split='train', transform=preprocess)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = dataSetup.ImageNette(root=file_path, split='val', transform=preprocess)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader