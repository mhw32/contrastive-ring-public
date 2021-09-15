import os
import copy
import getpass
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torchvision import datasets


class CIFAR10(data.Dataset):

    def __init__(
            self, 
            root,
            train=True, 
            image_transforms=None, 
        ):
        super().__init__()
        self.dataset = datasets.cifar.CIFAR10(
            root, 
            train=train,
            download=True,
            transform=image_transforms,
        )

    def __getitem__(self, index):
        img_data, label = self.dataset.__getitem__(index)
        return (index, img_data.float(), label)

    def __len__(self):
        return len(self.dataset)


class CIFAR10TwoViews(CIFAR10):
    """Returns two positive views, not one."""

    def __getitem__(self, index):
        img_data, label = self.dataset.__getitem__(index)
        img2_data, _ = self.dataset.__getitem__(index)
        return index, img_data.float(), img2_data.float(), label
