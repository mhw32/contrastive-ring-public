import cv2
import random
import numpy as np
from PIL import ImageFilter
from torchvision import transforms
from src.datasets.cifar10 import CIFAR10, CIFAR10TwoViews
from src.datasets.cifar100 import CIFAR100, CIFAR100TwoViews
from src.datasets.stl10 import STL10, STL10TwoViews

DATASET = {
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'stl10': STL10,
    'cifar10_2views': CIFAR10TwoViews,
    'cifar100_2views': CIFAR100TwoViews,
    'stl10_2views': STL10TwoViews,
}


def get_datasets(root, dataset_name, 
                 mocov2_transforms=False,
                 mocov2_32x32_transforms=False, 
                 mocov2_64x64_transforms=False,
                 mocov2_96x96_transforms=False):
    """
    Master function for loading datasets and toggle between
    different image transformation.
    """
    if mocov2_transforms:
        train_transforms, test_transforms = \
            load_mocov2_transforms()
    elif mocov2_32x32_transforms:
        train_transforms, test_transforms = \
            load_mocov2_32x32_transforms()
    elif mocov2_64x64_transforms:
        train_transforms, test_transforms = \
            load_mocov2_64x64_transforms()
    elif mocov2_96x96_transforms:
        train_transforms, test_transforms = \
            load_mocov2_96x96_transforms()
    else:
        train_transforms, test_transforms = \
           load_default_transforms()

    train_dataset = DATASET[dataset_name](
        root,
        train=True,
        image_transforms=train_transforms,
    )
    val_dataset = DATASET[dataset_name](
        root,
        train=False,
        image_transforms=test_transforms,
    )
    return train_dataset, val_dataset


def load_mocov2_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms, test_transforms


def load_mocov2_96x96_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=64, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms, test_transforms


def load_mocov2_64x64_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(73),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms, test_transforms


def load_mocov2_32x32_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms, test_transforms


def load_default_transforms():
    # resize imagenet to 256
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms, test_transforms


class GaussianBlur(object):

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
