import csv
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

DATA_DIR = Path("./unified_dataset")
METADATA_CSV = DATA_DIR / "metadata.csv"
IMG_SIZE = 224
MAX_PER_CLASS = 1000
BATCH_SIZE = 32
SEED = 42

# ImageNet normalization stats (standard for pretrained backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_metadata(metadata_csv=METADATA_CSV):
    """Load the unified metadata CSV and return list of row dicts."""
    rows = []
    with open(metadata_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_file_list(data_dir=DATA_DIR, metadata_csv=METADATA_CSV,
                    label_type="country", max_per_class=None,
                    min_per_class=20):
    """Build file list from the unified metadata CSV.

    Args:
        data_dir: Root directory containing the images.
        metadata_csv: Path to metadata.csv.
        label_type: "country" or "continent" — which column to use as class label.
        max_per_class: Cap images per class (handles imbalance).
        min_per_class: Drop classes with fewer than this many images.

    Returns:
        (file_paths, labels, class_names)
    """
    rows = load_metadata(metadata_csv)

    # Count per class and filter by minimum
    class_counts = Counter(row[label_type] for row in rows)
    valid_classes = {c for c, n in class_counts.items() if n >= min_per_class}

    # Group rows by class
    class_to_rows = {}
    for row in rows:
        cls = row[label_type]
        if cls in valid_classes:
            class_to_rows.setdefault(cls, []).append(row)

    # Sorted class names for deterministic indices
    class_names = sorted(class_to_rows.keys())
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    paths, labels = [], []
    for name in class_names:
        class_rows = sorted(class_to_rows[name], key=lambda r: r["filename"])
        if max_per_class and len(class_rows) > max_per_class:
            random.seed(SEED)
            class_rows = random.sample(class_rows, max_per_class)
        for row in class_rows:
            paths.append(data_dir / row["filename"])
            labels.append(class_to_idx[name])

    return paths, labels, class_names


def train_val_test_split(paths, labels, val_ratio=0.15, test_ratio=0.15):
    """Stratified split into train/val/test sets."""
    random.seed(SEED)

    # Group indices by label
    label_to_indices = {}
    for i, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    for label, indices in label_to_indices.items():
        random.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test:n_test + n_val])
        train_idx.extend(indices[n_test + n_val:])

    return train_idx, val_idx, test_idx


class GeoGuessrDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def make_weighted_sampler(labels):
    """Create a WeightedRandomSampler to balance class frequencies during training."""
    counts = Counter(labels)
    weight_per_class = {c: 1.0 / n for c, n in counts.items()}
    sample_weights = [weight_per_class[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))


def get_dataloaders(data_dir=DATA_DIR, metadata_csv=METADATA_CSV,
                    label_type="country", max_per_class=MAX_PER_CLASS,
                    min_per_class=20, batch_size=BATCH_SIZE, num_workers=4):
    """Build train/val/test DataLoaders with balancing and augmentation.

    Args:
        label_type: "country" (140 classes) or "continent" (7 classes).
    """
    paths, labels, class_names = build_file_list(
        data_dir, metadata_csv, label_type, max_per_class, min_per_class
    )
    train_idx, val_idx, test_idx = train_val_test_split(paths, labels)

    def subset(indices, transform):
        return GeoGuessrDataset(
            [paths[i] for i in indices],
            [labels[i] for i in indices],
            transform=transform,
        )

    train_ds = subset(train_idx, train_transform)
    val_ds = subset(val_idx, eval_transform)
    test_ds = subset(test_idx, eval_transform)

    sampler = make_weighted_sampler(train_ds.labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    for label_type in ["continent", "country"]:
        print(f"\n{'='*60}")
        print(f"Label type: {label_type}")
        print(f"{'='*60}")

        paths, labels, class_names = build_file_list(
            label_type=label_type, max_per_class=MAX_PER_CLASS, min_per_class=20
        )
        print(f"Classes: {len(class_names)}")
        print(f"Total images (capped at {MAX_PER_CLASS}/class, min {20}/class): {len(paths)}")

        train_idx, val_idx, test_idx = train_val_test_split(paths, labels)
        print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        print(f"\nClass distribution:")
        counts = Counter(labels)
        for i, name in enumerate(class_names):
            print(f"  {name:40s} {counts[i]:6d}")

    # Smoke test: load one batch (continent mode)
    print("\nLoading a batch (continent mode)...")
    train_loader, _, _, cnames = get_dataloaders(label_type="continent", num_workers=0)
    imgs, lbls = next(iter(train_loader))
    print(f"Batch shape: {imgs.shape}, labels: {lbls.shape}")
    print(f"Classes: {cnames}")
    print("Done.")
