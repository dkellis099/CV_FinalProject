import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import time

import hyperparameters as hp


BANNER_ID = 1865415 # <- must match student.py
torch.manual_seed(BANNER_ID)
np.random.seed(BANNER_ID)


# ========================================================================
#  SceneDataset — loads the 15-scenes dataset
#
class SceneDataset:
    """Load the 15-scenes dataset using ImageFolder (given, do not modify).

    Arguments:
        data_dir   -- path to dataset (must contain train/, val/, test/)
        batch_size -- batch size for DataLoaders
        image_size -- resize images to this square size

    After construction, provides:
        .train_loader  -- DataLoader for training set (shuffled)
        .val_loader    -- DataLoader for validation set
        .test_loader   -- DataLoader for test set
        .classes       -- list of class name strings
        .num_classes   -- number of classes
    """

    def __init__(self, data_dir, batch_size=hp.ENDTOEND_BATCH_SIZE, image_size=hp.ENDTOEND_IMAGE_SIZE):
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform)
        val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform)
        test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform)

        nw = 0 if os.name == 'nt' else 4
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=nw)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=nw)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=nw)
        self.classes = train_set.classes
        self.num_classes = len(self.classes)


# ========================================================================
#  Training loop
#
def train_loop(model, train_loader, optimizer, loss, epochs,
               device, val_loader=None, tasklabel="", on_epoch_end=None, frozen=False):
    """Train a model and optionally evaluate on a validation set each epoch.

    Arguments:
        model:          nn.Module to train
        train_loader:   DataLoader for training data
        optimizer:      torch.optim optimizer
        loss:           loss function (e.g., nn.CrossEntropyLoss())
        epochs:         number of training epochs
        device:         torch.device passed from main.py
        val_loader:     optional DataLoader for validation
        tasklabel:      string prefix for print output
        on_epoch_end:   optional callback, called as on_epoch_end(epoch, model)

    Returns:
        List of training accuracies     (float, one per epoch).
        List of validation accuracies   (float, one per epoch); empty if val_loader is None.
    """
    # model = model.to(device)
    train_accs = []
    val_accs = []
    start_time = time.time()
    current_time = start_time
    for epoch in range(epochs):
        # TODO: Implement the training loop. For each epoch:
        #     a. Set model to training mode.
        train_losses = []
        train_correct = 0
        train_size = len(train_loader.dataset)
        model.train()
        if frozen:
            model[0].eval()
        
        #     b. Loop over batches: move to device, forward pass, compute loss,
        #        backward pass, optimizer step. Track running accuracy and loss.
        for batch, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            
            pred = model(X)
            train_loss = loss(pred, y)
            train_losses.append(train_loss.item())
            train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            train_loss.backward()
            optimizer.step()
        train_acc = train_correct / train_size
        train_accs.append(train_acc)
        avg_loss = sum(train_losses) / len(train_losses)
        #     c. If val_loader is provided, evaluate: set model to eval mode,
        #        compute val accuracy with torch.no_grad(), append to val_accs.
        if val_loader is not None:
            model.eval()
            val_size = len(val_loader.dataset)
                            
            num_batches = len(val_loader)
            val_loss, num_correct = 0, 0
                
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(device)
                    y = y.to(device)
                    pred = model(X)
                    val_loss += loss(pred, y).item()
                    num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            val_loss /= num_batches
            val_acc = num_correct / val_size
            val_accs.append(val_acc)
        #     d. Print a status line each epoch (format shown below).
        #         f"[{tasklabel}] Epoch {epoch+1}/{epochs}  Train: {train_acc:.3f}  Loss: {avg_loss:.4f}"
        #         (append f"  Val: {val_acc:.3f}" if val_loader is provided)
        epoch_time = time.time()
        time_difference = epoch_time - current_time
        update = f"[{tasklabel}] Epoch {epoch+1}/{epochs}  Train: {train_acc:.3f}  Loss: {avg_loss:.4f}"
        if val_loader is not None:
            update += f"  Val: {val_acc:.3f}"
        update += f"  Time: {time_difference:.3f}"
        print(update)
        current_time = epoch_time

        #     e. If on_epoch_end is not None, call it at the end of an epoch: 
        #         on_epoch_end(epoch, model)
        
        if on_epoch_end is not None:
            on_epoch_end(epoch, model)
    end_time = time.time()
    total_time = end_time - start_time
    average_time = total_time / epochs
    print(f"[{tasklabel}] Total Time: {total_time:.3f}  Average Time: {average_time:.4f}")

    return train_accs, val_accs


# ========================================================================
#  CropRotationDataset — generates random rotated crops
#
class CropRotationDataset(Dataset):
    """Create a dataset of random rotated crops from images.

    Arguments:
        device     -- torch device for GPU-accelerated augmentation
        data_dir   -- path to a directory of images (with or without class subfolders)
        num_crops  -- total number of crops to generate per epoch
        crop_size  -- spatial size of each crop
        rotation   -- if True (default), apply random rotation and return rotation label
        batch_size -- batch size for the DataLoader

    After construction, provides:
        .train_loader  -- DataLoader for this dataset (shuffled)
        .classes       -- list of class name strings
        .num_classes   -- number of classes
    """

    def __init__(self, data_dir, device=None, num_crops=hp.ROTATION_NUM_CROPS,
                 crop_size=hp.ROTATION_CROP_SIZE, rotation=True,
                 batch_size=hp.ROTATION_BATCH_SIZE): 
        # TODO:
        entries = os.listdir(data_dir)

        subfolders = [
            name for name in entries
            if os.path.isdir(os.path.join(data_dir, name))
        ]
        # 1. Set self.num_crops, self.crop_size, self.rotation, self.batch_size
        #
        self.num_crops = num_crops
        self.crop_size = crop_size
        self.rotation = rotation
        self.batch_size = batch_size
        # 2. Set self.classes and self.num_classes.
        #    rotation=True  -> num_classes = 4 (one per rotation)
        #    rotation=False -> num_classes = number of class subfolders
        if rotation:
            self.num_classes = 4
        else:
            self.num_classes = len(subfolders)
        # 3. Load source images and transfer them to device as tensors.
        #    Note: Most datasets are too large to load all at once.
        #    We have a tiny dataset — just one or two images. So, it's ok.
        #
        to_tensor = transforms.ToTensor()

        self.images = []
        self.labels = []
        if len(subfolders) == 0:
            self.classes = range(4)
            for name in os.listdir(data_dir):
                path = os.path.join(data_dir, name)
                image = Image.open(path)
                image = to_tensor(image)
                self.images.append(image)
        else:
            self.classes = ['Street', 'Coast']
            class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

            for cls in self.classes:
                class_dir = os.path.join(data_dir, cls)
                for name in os.listdir(class_dir):
                    path = os.path.join(class_dir, name)
                    image = Image.open(path)
                    image = to_tensor(image)
                    self.images.append(image)
                    self.labels.append(class_to_idx[cls])
        # 4. Wrap this Dataset in a DataLoader for batching/shuffling:
        self.train_loader = DataLoader(self, batch_size=batch_size,
                                         shuffle=True, num_workers=0)

        # raise NotImplementedError("TODO: implement CropRotationDataset.__init__")

    def __len__(self):
        return self.num_crops

    def __getitem__(self, idx):
        """Return a random crop from a random source image.

        Returns:
            crop  -- (3, crop_size, crop_size) float32 tensor in [0, 1]
            label -- if rotation=True:  integer in {0, 1, 2, 3} (rotation class)
                     [Extra Credit] if rotation=False: integer class index {0, 1} (which directory, Street or Coast)
        """
        # TODO:
        # 1. Pick a random source image (as a tensor, already on device).
        index = np.random.choice(range(len(self.images)))
        image = self.images[index]
        # 2. Extract a random crop and rotate it at random as needed.
        _, h, w = image.shape
        top = np.random.choice(range(h - self.crop_size+1))
        left = np.random.choice(range(w - self.crop_size+1))   
        crop = transforms.functional.crop(image, top, left, self.crop_size, self.crop_size)
        
        if self.rotation:
            k = np.random.choice(range(4))
            label = k
            rotated = torch.rot90(crop, k, dims=(1, 2))
        else:
            rotated = crop
            label = self.labels[index]
            
        # 3. Add any other augmentations that might help.
        # 4. Define the label.
        return rotated, int(label)

        raise NotImplementedError("TODO: implement CropRotationDataset.__getitem__")
