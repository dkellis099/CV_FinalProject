import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


SEED = 1865415


def split_sun_csv(csv_path, val_ratio=0.15, test_ratio=0.15, seed=SEED):
    """Read the SUN attributes CSV and return train/val/test DataFrames."""
    df = pd.read_csv(csv_path)
    train_df, temp_df = train_test_split(df, test_size=val_ratio + test_ratio, random_state=seed)
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(temp_df, test_size=relative_test, random_state=seed)
    return train_df, val_df, test_df


class SUNAttributeDataset(Dataset):
    """PyTorch Dataset for SUN Attributes (102 continuous attribute labels).

    Arguments:
        dataframe   -- pandas DataFrame with 'image_path' and 102 attribute columns
        image_root  -- root directory containing the scene images
        transform   -- torchvision transform to apply to each image
    """

    def __init__(self, dataframe, image_root, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.image_paths = dataframe['image_path'].values
        self.labels = dataframe.drop(columns=['image_path']).values.astype(np.float32)
        self.num_attributes = self.labels.shape[1]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_root, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def get_sun_dataloaders(csv_path, image_root, batch_size=32, image_size=224):
    """Build train/val/test DataLoaders for SUN Attributes.

    Arguments:
        csv_path    -- path to sun_attributes.csv
        image_root  -- path to the images directory (e.g. sun_attributes/images)
        batch_size  -- batch size for all loaders
        image_size  -- resize images to this square size

    Returns:
        train_loader, val_loader, test_loader
    """
    train_df, val_df, test_df = split_sun_csv(csv_path)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    nw = 0 if os.name == 'nt' else 4

    train_set = SUNAttributeDataset(train_df, image_root, transform=train_transform)
    val_set = SUNAttributeDataset(val_df, image_root, transform=eval_transform)
    test_set = SUNAttributeDataset(test_df, image_root, transform=eval_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=nw)

    return train_loader, val_loader, test_loader
