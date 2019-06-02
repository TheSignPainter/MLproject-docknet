import torch
import torch.utils.data as data
from torchvision import transforms

import numpy as np
from PIL import Image
from pathlib import Path


class DockDataset(data.Dataset):
    def __init__(self, featdir='data/train', is_train=True, shuffle=True):
        self.data_path = []
        self.target = []

        # An ImageNet style transform
        self.transforms = transforms.Compose([
            # transforms.RandomSizedCrop(max(resize)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        base_path = Path(featdir)

        for p in base_path.glob('docks/*.jpg'):
            self.data_path.append(p)
            self.target.append(1)
        for p in base_path.glob('notdocks/*.jpg'):
            self.data_path.append(p)
            self.target.append(0)

        self.data_path = np.array(self.data_path)
        self.target = np.array(self.target)
        assert self.data_path.shape[0] == self.target.shape[0]

        if shuffle:
            index = np.arange(len(self.data_path))
            np.random.shuffle(index)
            self.data_path = self.data_path[index]
            self.target = self.target[index]
    
    def __getitem__(self, index):
        # Lazy, but memory efficient
        data = self.transforms(Image.open(self.data_path[index])).float()
        target = torch.LongTensor([self.target[index]])
        return data, target

    def __len__(self):
        return self.data_path.shape[0]