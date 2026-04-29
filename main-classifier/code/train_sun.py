import os
import numpy as np
import torch
import torch.nn as nn
import time

from sun_dataset import get_sun_dataloaders
from encoders import GeoEncoder
from helpers import load_dinov3_encoder, load_clip_encoder, load_convnext_encoder

import hyperparameters as hp


# ---------- Toggle which encoders to train ----------
run_dinov3 = 1
run_clip = 1
run_convnext = 1

# ---------- Paths ----------
CSV_PATH = '../../sun_attributes/sun_attributes.csv'
IMAGE_ROOT = '../../sun_attributes/images'
RESULTS_DIR = 'results/sun_attributes'

NUM_ATTRIBUTES = 102


def train_sun_loop(model, train_loader, optimizer, loss_fn, epochs, device,
                   val_loader=None, tasklabel=""):
    """Training loop for multi-label attribute regression."""
    train_losses = []
    val_losses = []
    start_time = time.time()
    current_time = start_time

    for epoch in range(epochs):
        model.train()
        model.encoder.eval()  # keep frozen encoder in eval mode

        epoch_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        status = f"[{tasklabel}] Epoch {epoch+1}/{epochs}  Train Loss: {avg_train_loss:.4f}"
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    preds = model(images)
                    val_loss += loss_fn(preds, labels).item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            status += f"  Val Loss: {avg_val_loss:.4f}"

        epoch_time = time.time()
        status += f"  Time: {epoch_time - current_time:.1f}s"
        print(status)
        current_time = epoch_time

    total_time = time.time() - start_time
    print(f"[{tasklabel}] Total Time: {total_time:.1f}s")

    return train_losses, val_losses


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    print("Loading SUN Attributes dataset...")
    train_loader, val_loader, test_loader = get_sun_dataloaders(
        CSV_PATH, IMAGE_ROOT, batch_size=hp.TRANSFER_BATCH_SIZE,
        image_size=hp.TRANSFER_IMAGE_SIZE,
    )
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}")

    loss_fn = nn.BCEWithLogitsLoss()

    # --- Frozen DINOv3 ---
    if run_dinov3:
        print("\n=== Frozen DINOv3 — SUN Attributes ===")
        encoder, embed_dim = load_dinov3_encoder(device=device)
        model = GeoEncoder(nn.Linear(embed_dim, NUM_ATTRIBUTES), encoder=encoder).to(device)
        optimizer = torch.optim.Adam(model.head.parameters(), lr=hp.TRANSFER_HEAD_LR)

        train_losses, val_losses = train_sun_loop(
            model, train_loader, optimizer, loss_fn,
            hp.TRANSFER_EPOCHS, device,
            val_loader=val_loader, tasklabel="DINOv3",
        )

        np.save(os.path.join(RESULTS_DIR, 'train_dinov3.npy'), train_losses)
        np.save(os.path.join(RESULTS_DIR, 'val_dinov3.npy'), val_losses)
        torch.save(model.head.state_dict(), os.path.join(RESULTS_DIR, 'dinov3_head.pt'))
        print("Saved DINOv3 results.")

    # --- Frozen CLIP ---
    if run_clip:
        print("\n=== Frozen CLIP — SUN Attributes ===")
        encoder, embed_dim = load_clip_encoder(device=device)
        model = GeoEncoder(nn.Linear(embed_dim, NUM_ATTRIBUTES), encoder=encoder).to(device)
        optimizer = torch.optim.Adam(model.head.parameters(), lr=hp.TRANSFER_HEAD_LR)

        train_losses, val_losses = train_sun_loop(
            model, train_loader, optimizer, loss_fn,
            hp.TRANSFER_EPOCHS, device,
            val_loader=val_loader, tasklabel="CLIP",
        )

        np.save(os.path.join(RESULTS_DIR, 'train_clip.npy'), train_losses)
        np.save(os.path.join(RESULTS_DIR, 'val_clip.npy'), val_losses)
        torch.save(model.head.state_dict(), os.path.join(RESULTS_DIR, 'clip_head.pt'))
        print("Saved CLIP results.")

    # --- Frozen ConvNext ---
    if run_convnext:
        print("\n=== Frozen ConvNext — SUN Attributes ===")
        encoder, embed_dim = load_convnext_encoder(device=device)
        model = GeoEncoder(nn.Linear(embed_dim, NUM_ATTRIBUTES), encoder=encoder).to(device)
        optimizer = torch.optim.Adam(model.head.parameters(), lr=hp.TRANSFER_HEAD_LR)

        train_losses, val_losses = train_sun_loop(
            model, train_loader, optimizer, loss_fn,
            hp.TRANSFER_EPOCHS, device,
            val_loader=val_loader, tasklabel="ConvNext",
        )

        np.save(os.path.join(RESULTS_DIR, 'train_convnext.npy'), train_losses)
        np.save(os.path.join(RESULTS_DIR, 'val_convnext.npy'), val_losses)
        torch.save(model.head.state_dict(), os.path.join(RESULTS_DIR, 'convnext_head.pt'))
        print("Saved ConvNext results.")

    print("\nDone! Results saved to", RESULTS_DIR)


if __name__ == '__main__':
    main()
