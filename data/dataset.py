# src/utils/dataset.py
"""
Data loading, transforms, and stratified train/val/test split.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split


def get_transforms(image_size: int = 224, crop_size: int = 200):
    """Return train and test transforms using ImageNet normalization."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    return train_transform, test_transform


def build_dataloaders(
        data_dir: str,
        batch_size: int = 16,
        num_workers: int = 2,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_state: int = 42,
        image_size: int = 224,
        crop_size: int = 200,
):
    """
    Load the dataset and return stratified train/val/test DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, calib_loader, class_names, test_idx
    """
    train_transform, test_transform = get_transforms(image_size, crop_size)

    # Load full dataset once to extract targets and indices
    full_dataset = ImageFolder(data_dir)
    targets = np.array(full_dataset.targets)
    indices = np.arange(len(targets))

    # Stratified split: train / (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_split + test_split),
        stratify=targets,
        random_state=random_state,
    )

    # Relative test size within the temp split
    relative_test_size = test_split / (val_split + test_split)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_size,
        stratify=targets[temp_idx],
        random_state=random_state,
    )

    # Build subsets with appropriate transforms
    train_dataset = torch.utils.data.Subset(ImageFolder(data_dir, transform=train_transform), train_idx)
    val_dataset   = torch.utils.data.Subset(ImageFolder(data_dir, transform=test_transform),  val_idx)
    test_dataset  = torch.utils.data.Subset(ImageFolder(data_dir, transform=test_transform),  test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    calib_loader = DataLoader(train_dataset, batch_size=32,         shuffle=True,  num_workers=num_workers)

    class_names = full_dataset.classes

    print(f"Dataset: {data_dir}")
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, calib_loader, train_dataset, class_names, test_idx