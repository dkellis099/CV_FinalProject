import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import time

import hyperparameters as hp


SEED = 1865415
torch.manual_seed(SEED)
np.random.seed(SEED)

# ========================================================================
#  SceneDataset — loads an image dataset
#
class SceneDataset:
    """Load an image dataset

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

    def __init__(self, data_dir, batch_size=hp.BATCH_SIZE, image_size=hp.IMAGE_SIZE):
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

    train_accs = []
    val_accs = []
    start_time = time.time()
    current_time = start_time
    for epoch in range(epochs):

        train_losses = []
        train_correct = 0
        train_size = len(train_loader.dataset)
        model.train()
        if frozen:
            model[0].eval()

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

        epoch_time = time.time()
        time_difference = epoch_time - current_time
        update = f"[{tasklabel}] Epoch {epoch+1}/{epochs}  Train: {train_acc:.3f}  Loss: {avg_loss:.4f}"
        if val_loader is not None:
            update += f"  Val: {val_acc:.3f}"
        update += f"  Time: {time_difference:.3f}"
        print(update)
        current_time = epoch_time
        
        if on_epoch_end is not None:
            on_epoch_end(epoch, model)
    end_time = time.time()
    total_time = end_time - start_time
    average_time = total_time / epochs
    print(f"[{tasklabel}] Total Time: {total_time:.3f}  Average Time: {average_time:.4f}")

    return train_accs, val_accs