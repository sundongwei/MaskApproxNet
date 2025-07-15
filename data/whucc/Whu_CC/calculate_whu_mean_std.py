import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from imageio import imread

class WHUCCDataset(Dataset):
    def __init__(self, data_folder, list_path, split, max_iters=None):
        self.list_path = list_path
        self.split = split

        assert self.split in {'train', 'val', 'test'}
        self.img_ids = [i_id.strip() for i_id in open(os.path.join(list_path, split + '.txt'))]
        if max_iters is not None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters - n_repeat * len(self.img_ids)]
        self.files = []
        for name in self.img_ids:
            img_fileA = os.path.join(data_folder, split, 'A', name.split('-')[0])
            img_fileB = img_fileA.replace('A', 'B')
            self.files.append({
                "imgA": img_fileA,
                "imgB": img_fileB,
                "name": name.split('-')[0]
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_file = self.files[index]
        imgA = imread(data_file["imgA"])
        imgB = imread(data_file["imgB"])

        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)

        imgA = np.moveaxis(imgA, -1, 0)
        imgB = np.moveaxis(imgB, -1, 0)

        return imgA, imgB

def calculate_mean_std(data_folder, list_path, split, batch_size=128):
    dataset = WHUCCDataset(data_folder, list_path, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    channels_sumA, channels_sumB = 0, 0
    channels_squared_sumA, channels_squared_sumB = 0, 0
    num_batches = len(loader)

    for imgA, imgB in loader:
        channels_sumA += torch.mean(imgA, dim=[0, 2, 3])
        channels_squared_sumA += torch.mean(imgA ** 2, dim=[0, 2, 3])
        channels_sumB += torch.mean(imgB, dim=[0, 2, 3])
        channels_squared_sumB += torch.mean(imgB ** 2, dim=[0, 2, 3])

    meanA = channels_sumA / num_batches
    meanB = channels_sumB / num_batches

    stdA = (channels_squared_sumA / num_batches - meanA ** 2) ** 0.5
    stdB = (channels_squared_sumB / num_batches - meanB ** 2) ** 0.5

    return meanA, stdA, meanB, stdB

if __name__ == "__main__":
    data_folder = './data/Whu_CC/whu_CDC_dataset/images'
    list_path = './data/Whu_CC/'
    split = 'train'
    meanA, stdA, meanB, stdB = calculate_mean_std(data_folder, list_path, split)
    print(f"Mean A: {meanA}, Std A: {stdA}")
    print(f"Mean B: {meanB}, Std B: {stdB}")
    print(f"Mean: {(meanA + meanB) / 2}, Std: {(stdA + stdB) / 2}")
