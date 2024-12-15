from recipe.training import Training
import torch
from datasets.cifar10 import cifar10
from datasets.cifar100 import cifar100
from datasets.cinic10 import cinic10
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='It is the simulation of VFL with SplitNN')
    parser.add_argument('-n', '--num_client', default=2, type=int, help='The number of clients in the Simulation')
    parser.add_argument('-d', '--dataset', default='cifar10', type=str, help='The dataset used in the Simulation')
    parser.add_argument('-e', '--epochs', default=20, type=int, help='The epochs of training')
    parser.add_argument('-b', '--batch_size', default=128, type=int, help='The batch size of training')
    args = parser.parse_args()
    num_clients = args.num_client
    epochs = args.epochs
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == 'cifar10':
        file_path = cifar10()
        train = Training(num_clients, file_path, epochs, batch_size, device)
        train.trainingCIFAR10()
    elif args.dataset == 'cifar100':
        file_path = cifar100()
        train = Training(num_clients, file_path, epochs, batch_size, device)
        train.trainingCIFAR100()
    elif args.dataset == 'cinic10':
        file_path = cinic10()
        train = Training(num_clients, file_path, epochs, batch_size, device)
        train.trainingCINIC10()
    
    
    
