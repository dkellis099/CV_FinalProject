import os
import numpy as np
import torch
import torch.nn as nn

from sun_dataset import get_sun_dataloaders
from encoders import GeoEncoder
from helpers import load_dinov3_encoder, load_clip_encoder, load_convnext_encoder

import hyperparameters as hp


CSV_PATH = '../../sun_attributes/sun_attributes.csv'
IMAGE_ROOT = '../../sun_attributes/images'
RESULTS_DIR = 'results/sun_attributes'

NUM_ATTRIBUTES = 102

# Encoder name -> (loader function, head weights file)
ENCODERS = {
    'DINOv3':  (load_dinov3_encoder, 'dinov3_head.pt'),
    'CLIP':    (load_clip_encoder,   'clip_head.pt'),
    'ConvNext': (load_convnext_encoder, 'convnext_head.pt'),
}


def evaluate(model, test_loader, device):
    """Run model on test set, return per-attribute MSE."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = torch.sigmoid(model(images)).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # MSE per attribute
    per_attr_mse = ((all_preds - all_labels) ** 2).mean(axis=0)
    overall_mse = per_attr_mse.mean()

    return overall_mse, per_attr_mse


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test set
    _, _, test_loader = get_sun_dataloaders(
        CSV_PATH, IMAGE_ROOT, batch_size=hp.TRANSFER_BATCH_SIZE,
        image_size=hp.TRANSFER_IMAGE_SIZE,
    )
    print(f"Test set: {len(test_loader.dataset)} images\n")

    # Get attribute names from CSV header
    import pandas as pd
    df = pd.read_csv(CSV_PATH, nrows=0)
    attr_names = [c for c in df.columns if c != 'image_path']

    results = {}

    for name, (loader_fn, weights_file) in ENCODERS.items():
        weights_path = os.path.join(RESULTS_DIR, weights_file)
        if not os.path.exists(weights_path):
            print(f"Skipping {name} — {weights_file} not found")
            continue

        print(f"=== Evaluating {name} ===")
        encoder, embed_dim = loader_fn(device=device)
        head = nn.Linear(embed_dim, NUM_ATTRIBUTES)
        head.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model = GeoEncoder(head=head, encoder=encoder).to(device)

        overall_mse, per_attr_mse = evaluate(model, test_loader, device)
        results[name] = (overall_mse, per_attr_mse)

        # Top 5 best and worst attributes
        sorted_idx = np.argsort(per_attr_mse)
        best_5 = sorted_idx[:5]
        worst_5 = sorted_idx[-5:][::-1]

        print(f"  Overall MSE: {overall_mse:.4f}")
        print(f"  Best 5 attributes:")
        for i in best_5:
            print(f"    {attr_names[i]:40s} MSE: {per_attr_mse[i]:.4f}")
        print(f"  Worst 5 attributes:")
        for i in worst_5:
            print(f"    {attr_names[i]:40s} MSE: {per_attr_mse[i]:.4f}")
        print()

        # Save per-attribute MSE
        np.save(os.path.join(RESULTS_DIR, f'mse_{name.lower()}.npy'), per_attr_mse)

    # Print comparison table
    if results:
        print("=" * 60)
        print(f"{'Encoder':<12} {'Overall MSE':<15} {'Best Attr (MSE)':<30} {'Worst Attr (MSE)'}")
        print("-" * 60)
        for name, (overall, per_attr) in results.items():
            best_i = np.argmin(per_attr)
            worst_i = np.argmax(per_attr)
            print(f"{name:<12} {overall:<15.4f} {attr_names[best_i]} ({per_attr[best_i]:.4f})   {attr_names[worst_i]} ({per_attr[worst_i]:.4f})")
        print("=" * 60)


if __name__ == '__main__':
    main()
