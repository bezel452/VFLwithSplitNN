from recipe.training import Training
import torch
from datasets.cifar10 import cifar10

if __name__ == '__main__':
    num_clients = 2
    file_path = cifar10()
    print(file_path)
    epochs = 20
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = Training(num_clients, file_path, epochs, batch_size, device)
    train.trainingCIFAR10()