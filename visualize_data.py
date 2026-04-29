import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from DataProcessing import (DATA_DIR, METADATA_CSV, build_file_list,
                             load_metadata, MAX_PER_CLASS, SEED)

random.seed(SEED)

# Load raw metadata for source breakdown
raw_meta = load_metadata(METADATA_CSV)

# --- 1. Continent distribution ---
continent_counts = Counter(r["continent"] for r in raw_meta)
fig, ax = plt.subplots(figsize=(10, 5))
continents = sorted(continent_counts.keys(), key=lambda c: -continent_counts[c])
values = [continent_counts[c] for c in continents]
colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1"]
ax.bar(continents, values, color=colors[:len(continents)])
ax.set_ylabel("Number of Images")
ax.set_title(f"Unified Dataset by Continent — {len(raw_meta):,} total images")
for i, (c, v) in enumerate(zip(continents, values)):
    ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("continent_distribution.png", dpi=150)
plt.show()

# --- 2. Source breakdown ---
source_counts = Counter(r["source"] for r in raw_meta)
fig, ax = plt.subplots(figsize=(7, 5))
sources = sorted(source_counts.keys(), key=lambda s: -source_counts[s])
src_values = [source_counts[s] for s in sources]
ax.bar(sources, src_values, color=["#4e79a7", "#f28e2b", "#e15759"])
ax.set_ylabel("Number of Images")
ax.set_title("Images by Source Dataset")
for i, v in enumerate(src_values):
    ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("source_distribution.png", dpi=150)
plt.show()

# --- 3. Country distribution (top 40 + bottom 10) ---
paths, labels, class_names = build_file_list(
    label_type="country", max_per_class=MAX_PER_CLASS, min_per_class=20
)
counts = Counter(labels)
sorted_classes = sorted(range(len(class_names)), key=lambda i: -counts[i])

# Top 40
top_n = 40
top_idx = sorted_classes[:top_n]
top_names = [class_names[i] for i in top_idx]
top_values = [counts[i] for i in top_idx]

fig, ax = plt.subplots(figsize=(14, 10))
ax.barh(top_names, top_values, color="steelblue")
ax.set_xlabel("Number of Images")
ax.set_title(f"Top {top_n} Countries (capped at {MAX_PER_CLASS}/class, min 20/class) — "
             f"{len(class_names)} countries, {len(paths):,} images")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("country_distribution.png", dpi=150)
plt.show()

# --- 4. Sample images grid: 1 image from each continent ---
continent_paths = {}
for row in raw_meta:
    c = row["continent"]
    continent_paths.setdefault(c, []).append(DATA_DIR / row["filename"])

display_continents = [c for c in continents if len(continent_paths.get(c, [])) >= 2
                      and c != "Antarctica"]
n_cols = len(display_continents)
fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))

for col, continent in enumerate(display_continents):
    samples = random.sample(continent_paths[continent], 2)
    for row, img_path in enumerate(samples):
        img = Image.open(img_path).convert("RGB")
        axes[row, col].imshow(img)
        axes[row, col].set_title(continent, fontsize=10, fontweight="bold")
        axes[row, col].axis("off")

plt.suptitle("Sample Images by Continent", fontsize=14)
plt.tight_layout()
plt.savefig("sample_images.png", dpi=150)
plt.show()

# --- 5. Source overlap per continent ---
fig, ax = plt.subplots(figsize=(12, 6))
source_names = sorted(source_counts.keys())
continent_source = {}
for r in raw_meta:
    key = (r["continent"], r["source"])
    continent_source[key] = continent_source.get(key, 0) + 1

x = range(len(display_continents))
width = 0.25
for i, src in enumerate(source_names):
    vals = [continent_source.get((c, src), 0) for c in display_continents]
    ax.bar([xi + i * width for xi in x], vals, width, label=src)

ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels(display_continents)
ax.set_ylabel("Number of Images")
ax.set_title("Source Contribution by Continent")
ax.legend()
plt.tight_layout()
plt.savefig("source_by_continent.png", dpi=150)
plt.show()

print("Saved: continent_distribution.png, source_distribution.png, "
      "country_distribution.png, sample_images.png, source_by_continent.png")
