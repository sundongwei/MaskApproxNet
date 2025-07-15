import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class whuDataset(Dataset):
    def __init__(self, root_dir='./data/whu/Whu_CC/whu_CDC_dataset', split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of dataset
            split (str): train/val/test
            transform (callable, optional): Transform for remote sensing images
        """
        self.root_dir = root_dir
        self.split = split
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

        # Set up paths for bi-temporal images and change map
        self.imageA_dir = os.path.join(root_dir, 'images', split, 'A')  # T1 time images
        self.imageB_dir = os.path.join(root_dir, 'images', split, 'B')  # T2 time images
        self.label_dir = os.path.join(root_dir, 'images', split, 'label')    # Change map
        
        # Get image file list
        self.image_files = sorted(os.listdir(self.imageA_dir))

        # Label transform for change map
        self.label_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load bi-temporal images
        img_name = self.image_files[idx]
        imageA_path = os.path.join(self.imageA_dir, img_name)
        imageB_path = os.path.join(self.imageB_dir, img_name)
        
        imageA = Image.open(imageA_path).convert('RGB')  # T1 image
        imageB = Image.open(imageB_path).convert('RGB')  # T2 image

        if self.transform:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)

        # Load change map for train and val splits
        if self.split in ['train', 'val']:
            label_path = os.path.join(self.label_dir, img_name)
            label = Image.open(label_path).convert('L')
            label = self.label_transform(label)
            label = (label > 0.5).float()  # Binary change map
            
            return imageA, imageB, label  # Binary change map
        
        return imageA, imageB, str(img_name)  # Test split

if __name__ == "__main__":
    # Test the WHU dataset
    ds = whuDataset(root_dir='./data/whu/Whu_CC/whu_CDC_dataset', split='test')
    for i in range(len(ds)):
        imageA, imageB, img_name = ds[i]
        print(imageA.shape, imageB.shape, img_name)
        print(os.path.splitext(img_name)[0])
        print("--------")