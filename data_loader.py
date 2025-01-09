import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from glob import glob
import os
from PIL import Image 
import numpy as np


# Create a custom dataset class
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, image_size, is_train=True):
        self.image_paths = image_paths

        if is_train:
            self.transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                                # transforms.RandomVerticalFlip(),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),])
        else:
            self.transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                                transforms.ToTensor(),])
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        output = {}
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Load image and ensure RGB format
        if self.transform:
            image = self.transform(image)
        mask = ~torch.all(image == image[0][0][0], dim=0, keepdim=True)
        output['masks'] = mask
        output['images'] = (image * 2 - 1) * mask
        return output

    
