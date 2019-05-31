import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import sys, os
from pathlib import Path


class EGGDataset(data.Dataset):
    def __init__(self, featdir='data/train', is_train=True, shuffle=True):
        self.data_tensor = []
        self.target_tensor = []

        base_path = Path(featdir)

        for p in base_path.glob('docks/*.jpg'):
            # 3 x H x W
            data = np.array(Image.open(p)).transpose(2,0,1)
            self.data_tensor.append(data)
            self.target_tensor.append(1)
        for p in base_path.glob('notdocks/*.jpg'):
            # 3 x H x W
            data = np.array(Image.open(p)).transpose(2,0,1)
            self.data_tensor.append(data)
            self.target_tensor.append(0)

        self.data_tensor = np.array(self.data_tensor)
        self.target_tensor = np.array(self.target_tensor)
        assert self.data_tensor.shape[0] == self.target_tensor.shape[0]

        if shuffle:
            index = np.arange(len(self.data_tensor))
            np.random.shuffle(index)
            self.data_tensor = self.data_tensor[index]
            self.target_tensor = self.target_tensor[index]
        
        self.data_tensor = torch.from_numpy(self.data_tensor).float()
        self.target_tensor = torch.from_numpy(self.target_tensor).long()
    
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]