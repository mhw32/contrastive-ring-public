import os
import copy
import getpass
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torchvision import datasets


class STL10(data.Dataset):

    def __init__(
            self, 
            root,
            train=True, 
            image_transforms=None, 
        ):
        super().__init__()
        split = 'train' if train else 'test'
        self.dataset = datasets.stl10.STL10(
            root, 
            split=split,
            download=True,
            transform=image_transforms,
        )
        self.dataset.targets = copy.copy(self.dataset.labels)

    def __getitem__(self, index):
        img_data, label = self.dataset.__getitem__(index)
        return (index, img_data.float(), label)

    def __len__(self):
        return len(self.dataset)


class STL10TwoViews(STL10):
    """Returns two positive views, not one."""

    def __getitem__(self, index):
        img_data, label = self.dataset.__getitem__(index)
        img2_data, _ = self.dataset.__getitem__(index)
        return index, img_data.float(), img2_data.float(), label
