import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CINIC10(Dataset):
    def __init__(self, root, split='train', transform=None):
        super().__init__()
        image_folder = torchvision.datasets.ImageFolder(root=root + '/' + split)
        self.targets = image_folder.targets
        self.image_paths = image_folder.imgs
        self.transform = transform

    def __getitem__(self, index):
        file_path, label = self.image_paths[index]
        img = self.read_image(file_path)
        return img, label
    
    def __len__(self):
        return len(self.image_paths)

    def read_image(self, path):
        img = Image.open(path)
        return self.transform(img) if self.transform else img
    
class ImageNette(Dataset):
    def __init__(self, root, split='train', transform=None):
        super().__init__()
        image_folder = torchvision.datasets.ImageFolder(root=root + '/' + split)
        self.targets = image_folder.targets
        self.image_paths = image_folder.imgs
        self.transform = transform

    def __getitem__(self, index):
        file_path, label = self.image_paths[index]
        img = self.read_image(file_path)
        return img, label
    
    def __len__(self):
        return len(self.image_paths)

    def read_image(self, path):
        img = Image.open(path)
        return self.transform(img) if self.transform else img
    
class BreastData(Dataset):
    def __init__(self, feature, label):
        self.feature, self.label = feature, label

    def __getitem__(self, index):
        return self.feature[index], self.label[index]
    
    def __len__(self):
        return len(self.label)