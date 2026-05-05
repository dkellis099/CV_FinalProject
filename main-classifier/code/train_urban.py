"""
Train urban/suburban/rural classification on the Global Streetscapes dataset.
Uses the same encoder + linear head pipeline as tasks.py.

Usage:
    python train_urban.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from encoders import GeoEncoder
from utils import train_loop
from helpers import load_clip_encoder

import hyperparameters as hp


# ---------- Toggle which encoders to train ----------
run_dinov3 = 0
run_clip_lora = 1
run_convnext = 0

# ---------- Paths ----------
PT_PATH = '../../sun_attributes/streetscapes_20k.pt'
RESULTS_DIR = 'results/urban'

NUM_CLASSES = 3
LABEL_NAMES = ['urban', 'suburban', 'rural']
SEED = 1865415


class PTDataset(Dataset):
    """Wraps image tensors and labels from a .pt file."""
    def __init__(self, images, labels, normalize=None):
        self.images = images
        self.labels = labels
        self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].float() / 255.0  # uint8 -> float [0,1]
        if self.normalize:
            img = self.normalize(img)
        return img, self.labels[idx]


def load_urban_data(pt_path, batch_size=hp.TRANSFER_BATCH_SIZE):
    """Load .pt file and create train/val/test DataLoaders."""
    from torchvision import transforms

    print(f"Loading {pt_path}...")
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    images = data['images']   # uint8 [N, 3, 224, 224]
    labels = data['labels']   # int64 [N]

    print(f"Loaded {len(labels)} images, {NUM_CLASSES} classes")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name}: {(labels == i).sum().item()}")

    # 70/15/15 split
    indices = list(range(len(labels)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=labels.numpy(), random_state=SEED)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=labels[temp_idx].numpy(), random_state=SEED)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_set = PTDataset(images[train_idx], labels[train_idx], normalize=normalize)
    val_set = PTDataset(images[val_idx], labels[val_idx], normalize=normalize)
    test_set = PTDataset(images[test_idx], labels[test_idx], normalize=normalize)

    nw = 0 if os.name == 'nt' else 4
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=nw)

    return train_loader, val_loader, test_loader


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_loader, val_loader, test_loader = load_urban_data(PT_PATH)
    loss_fn = nn.CrossEntropyLoss()

    # --- LoRA CLIP ---
    if run_clip_lora:
        print("\n=== LoRA CLIP — Urban Classification ===")
        encoder, embed_dim = load_clip_encoder(device=device)
        model = GeoEncoder(nn.Linear(embed_dim, NUM_CLASSES), encoder=encoder, lora=True).to(device)
        optimizer = torch.optim.Adam([
            {"params": model.head.parameters(), "lr": hp.TRANSFER_HEAD_LR},
            {"params": [p for n, p in model.encoder.named_parameters() if p.requires_grad], "lr": hp.LORA_LR},
        ])

        train_accs, val_accs = train_loop(
            model, train_loader, optimizer, loss_fn,
            hp.TRANSFER_EPOCHS, device,
            val_loader=val_loader, tasklabel="LoRA-CLIP-Urban")

        np.save(os.path.join(RESULTS_DIR, 'train_clip_lora.npy'), train_accs)
        np.save(os.path.join(RESULTS_DIR, 'val_clip_lora.npy'), val_accs)
        torch.save(model.head.state_dict(), os.path.join(RESULTS_DIR, 'clip_lora_head.pt'))
        torch.save(model.encoder.state_dict(), os.path.join(RESULTS_DIR, 'clip_lora_encoder.pt'))
        print("Saved CLIP LoRA results.")

    print(f"\nDone! Results saved to {RESULTS_DIR}")


if __name__ == '__main__':
    main()
