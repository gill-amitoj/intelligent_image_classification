# ----------------------------------------------------------
# Intelligent Image Classification System
# Author: Amitoj Singh (CCID: amitoj3)
# ----------------------------------------------------------

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import ImageFile

# Allow loading truncated images instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_data_loaders(train_dir, val_dir, batch_size=32, img_size=224):
    """Return PyTorch dataloaders for training and validation."""
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, len(train_dataset.classes)

