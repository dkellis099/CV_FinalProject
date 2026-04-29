import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sun_dataset import get_sun_dataloaders, split_sun_csv, SUNAttributeDataset
from encoders import GeoEncoder
from helpers import load_dinov3_encoder, load_clip_encoder, load_convnext_encoder
from torchvision import transforms

import hyperparameters as hp


CSV_PATH = '../../sun_attributes/sun_attributes.csv'
IMAGE_ROOT = '../../sun_attributes/images'
RESULTS_DIR = 'results/sun_attributes'

NUM_ATTRIBUTES = 102

# Maps encoder name -> (loader function, head weights file)
ENCODERS = {
    'dinov3':   (load_dinov3_encoder, 'dinov3_head.pt'),
    'clip':     (load_clip_encoder,   'clip_head.pt'),
    'convnext': (load_convnext_encoder, 'convnext_head.pt'),
}


def generate_embeddings(model, image_paths, image_root, device, batch_size=32, image_size=224):
    """Run model on all images and return 102-dim attribute predictions."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    # Minimal dataset just for loading images
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    class ImageOnlyDataset(Dataset):
        def __init__(self, paths, root, transform):
            self.paths = paths
            self.root = root
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(os.path.join(self.root, self.paths[idx])).convert('RGB')
            return self.transform(img)

    dataset = ImageOnlyDataset(image_paths, image_root, transform)
    nw = 0 if os.name == 'nt' else 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    model.eval()
    all_preds = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            preds = torch.sigmoid(model(images)).cpu()
            all_preds.append(preds)

    return torch.cat(all_preds, dim=0).numpy()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load all image paths from CSV
    df = pd.read_csv(CSV_PATH)
    image_paths = df['image_path'].values
    attr_names = [c for c in df.columns if c != 'image_path']
    print(f"Generating embeddings for {len(image_paths)} images\n")

    for enc_name, (loader_fn, weights_file) in ENCODERS.items():
        weights_path = os.path.join(RESULTS_DIR, weights_file)
        if not os.path.exists(weights_path):
            print(f"Skipping {enc_name} — {weights_file} not found")
            continue

        print(f"=== Generating embeddings with {enc_name} ===")
        encoder, embed_dim = loader_fn(device=device)
        head = nn.Linear(embed_dim, NUM_ATTRIBUTES)
        head.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model = GeoEncoder(head=head, encoder=encoder).to(device)

        embeddings = generate_embeddings(model, image_paths, IMAGE_ROOT, device)

        # Save as numpy array (14340 x 102)
        npy_path = os.path.join(RESULTS_DIR, f'embeddings_{enc_name}.npy')
        np.save(npy_path, embeddings)
        print(f"  Saved {embeddings.shape} to {npy_path}")

        # Save as CSV with image paths and attribute names
        csv_path = os.path.join(RESULTS_DIR, f'embeddings_{enc_name}.csv')
        emb_df = pd.DataFrame(embeddings, columns=attr_names)
        emb_df.insert(0, 'image_path', image_paths)
        emb_df.to_csv(csv_path, index=False)
        print(f"  Saved CSV to {csv_path}")
        print()

    print("Done!")


if __name__ == '__main__':
    main()
