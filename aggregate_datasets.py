"""
Aggregate three geolocation datasets into a single unified dataset.

Sources:
  1. geoguessr_filtered_data/   — annaglass1, 56 countries, folder-based labels
  2. geoguessr_50k_data/        — ubitquitin, 124 countries, folder-based labels
  3. google_street_view_data/   — paulchambaz, 106 countries, coords + reverse-geocoded labels

Output:
  - unified_dataset/            — images organized by country folders
  - unified_dataset/metadata.csv — master index with filename, country, continent, source
"""

import csv
import os
import shutil
from collections import Counter
from pathlib import Path

COUNTRY_TO_CONTINENT = {
    "Afghanistan": "Asia", "Aland": "Europe", "Albania": "Europe", "Algeria": "Africa",
    "American Samoa": "Oceania", "Andorra": "Europe", "Angola": "Africa", "Antarctica": "Antarctica",
    "Argentina": "South America", "Armenia": "Asia", "Australia": "Oceania", "Austria": "Europe",
    "Azerbaijan": "Asia", "Bangladesh": "Asia", "Belarus": "Europe", "Belgium": "Europe",
    "Belize": "North America", "Bermuda": "North America", "Bhutan": "Asia", "Bolivia": "South America",
    "Bosnia and Herzegovina": "Europe", "Botswana": "Africa", "Brazil": "South America",
    "Brunei": "Asia", "Bulgaria": "Europe", "Burkina Faso": "Africa", "Cambodia": "Asia",
    "Cameroon": "Africa", "Canada": "North America", "Cape Verde": "Africa",
    "Central African Republic": "Africa", "Chile": "South America", "China": "Asia",
    "Colombia": "South America", "Costa Rica": "North America", "Croatia": "Europe",
    "Cuba": "North America", "Curacao": "North America", "Cyprus": "Europe", "Czechia": "Europe",
    "DR Congo": "Africa", "Denmark": "Europe", "Dominican Republic": "North America",
    "Ecuador": "South America", "Egypt": "Africa", "El Salvador": "North America",
    "Equatorial Guinea": "Africa", "Eritrea": "Africa", "Estonia": "Europe", "Eswatini": "Africa",
    "Ethiopia": "Africa", "Faroe Islands": "Europe", "Fiji": "Oceania", "Finland": "Europe",
    "France": "Europe", "French Guiana": "South America", "Gabon": "Africa", "Gambia": "Africa",
    "Georgia": "Asia", "Germany": "Europe", "Ghana": "Africa", "Gibraltar": "Europe",
    "Greece": "Europe", "Greenland": "North America", "Guam": "Oceania",
    "Guatemala": "North America", "Guinea": "Africa", "Guinea-Bissau": "Africa",
    "Guyana": "South America", "Haiti": "North America", "Honduras": "North America",
    "Hong Kong": "Asia", "Hungary": "Europe", "Iceland": "Europe", "India": "Asia",
    "Indonesia": "Asia", "Iran": "Asia", "Iraq": "Asia", "Ireland": "Europe",
    "Isle of Man": "Europe", "Israel": "Asia", "Italy": "Europe", "Ivory Coast": "Africa",
    "Jamaica": "North America", "Japan": "Asia", "Jersey": "Europe", "Jordan": "Asia",
    "Kazakhstan": "Asia", "Kenya": "Africa", "Kyrgyzstan": "Asia", "Laos": "Asia",
    "Latvia": "Europe", "Lebanon": "Asia", "Lesotho": "Africa", "Liberia": "Africa",
    "Libya": "Africa", "Lithuania": "Europe", "Luxembourg": "Europe", "Macao": "Asia",
    "Madagascar": "Africa", "Malaysia": "Asia", "Mali": "Africa", "Malta": "Europe",
    "Martinique": "North America", "Mauritania": "Africa", "Mexico": "North America",
    "Moldova": "Europe", "Monaco": "Europe", "Mongolia": "Asia", "Montenegro": "Europe",
    "Morocco": "Africa", "Mozambique": "Africa", "Myanmar": "Asia", "Namibia": "Africa",
    "Nepal": "Asia", "Netherlands": "Europe", "New Zealand": "Oceania", "Nicaragua": "North America",
    "Niger": "Africa", "Nigeria": "Africa", "North Macedonia": "Europe",
    "Northern Mariana Islands": "Oceania", "Norway": "Europe", "Oman": "Asia",
    "Pakistan": "Asia", "Palestine": "Asia", "Panama": "North America", "Paraguay": "South America",
    "Peru": "South America", "Philippines": "Asia", "Pitcairn Islands": "Oceania",
    "Poland": "Europe", "Portugal": "Europe", "Puerto Rico": "North America", "Qatar": "Asia",
    "Reunion": "Africa", "Romania": "Europe", "Russia": "Europe", "Rwanda": "Africa",
    "San Marino": "Europe", "Saudi Arabia": "Asia", "Senegal": "Africa", "Serbia": "Europe",
    "Sierra Leone": "Africa", "Singapore": "Asia", "Slovakia": "Europe", "Slovenia": "Europe",
    "Somalia": "Africa", "South Africa": "Africa", "South Georgia and South Sandwich Islands": "Antarctica",
    "South Korea": "Asia", "South Sudan": "Africa", "Spain": "Europe", "Sri Lanka": "Asia",
    "Sudan": "Africa", "Svalbard and Jan Mayen": "Europe", "Sweden": "Europe",
    "Switzerland": "Europe", "Syria": "Asia", "Taiwan": "Asia", "Tajikistan": "Asia",
    "Tanzania": "Africa", "Thailand": "Asia", "Togo": "Africa", "Tunisia": "Africa",
    "Turkey": "Asia", "US Virgin Islands": "North America", "Uganda": "Africa",
    "Ukraine": "Europe", "United Arab Emirates": "Asia", "United Kingdom": "Europe",
    "United States": "North America", "Uruguay": "South America", "Uzbekistan": "Asia",
    "Vanuatu": "Oceania", "Venezuela": "South America", "Vietnam": "Asia",
    "Yemen": "Asia", "Zambia": "Africa", "Zimbabwe": "Africa",
}

OUTPUT_DIR = Path("unified_dataset")
SOURCES = {
    "annaglass1": Path("geoguessr_filtered_data"),
    "ubitquitin": Path("geoguessr_50k_data"),
    "paulchambaz": Path("google_street_view_data"),
}


def collect_folder_based(source_name, source_dir):
    """Collect entries from a dataset organized as country/image folders."""
    entries = []
    for country_dir in sorted(source_dir.iterdir()):
        if not country_dir.is_dir():
            continue
        country = country_dir.name
        continent = COUNTRY_TO_CONTINENT.get(country)
        if continent is None:
            print(f"  WARNING: no continent mapping for '{country}' in {source_name}, skipping")
            continue
        for img_path in sorted(country_dir.glob("*")):
            if img_path.is_file() and img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                entries.append({
                    "src_path": img_path,
                    "country": country,
                    "continent": continent,
                    "source": source_name,
                })
    return entries


def collect_paulchambaz(source_dir):
    """Collect entries from paulchambaz using the labels.csv we generated."""
    entries = []
    labels_path = source_dir / "labels.csv"
    with open(labels_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            country = row["country"]
            continent = COUNTRY_TO_CONTINENT.get(country)
            img_path = source_dir / row["filename"]
            if continent is None:
                print(f"  WARNING: no continent mapping for '{country}' in paulchambaz, skipping")
                continue
            if not img_path.exists():
                continue
            entries.append({
                "src_path": img_path,
                "country": country,
                "continent": continent,
                "source": "paulchambaz",
            })
    return entries


def main():
    all_entries = []

    # Folder-based datasets
    for name in ["annaglass1", "ubitquitin"]:
        print(f"Scanning {name}...")
        entries = collect_folder_based(name, SOURCES[name])
        print(f"  {len(entries)} images")
        all_entries.extend(entries)

    # Coordinate-based dataset
    print("Scanning paulchambaz...")
    entries = collect_paulchambaz(SOURCES["paulchambaz"])
    print(f"  {len(entries)} images")
    all_entries.extend(entries)

    print(f"\nTotal: {len(all_entries)} images")

    # Create output directory structure and copy images
    OUTPUT_DIR.mkdir(exist_ok=True)
    country_counters = Counter()
    metadata_rows = []

    print("Copying images to unified_dataset/...")
    for entry in all_entries:
        country = entry["country"]
        country_dir = OUTPUT_DIR / country
        country_dir.mkdir(exist_ok=True)

        # Unique filename: source_originalname
        src_stem = entry["src_path"].stem
        src_suffix = entry["src_path"].suffix
        new_name = f"{entry['source']}_{src_stem}{src_suffix}"
        dst_path = country_dir / new_name

        # Handle rare collisions
        if dst_path.exists():
            country_counters[country] += 1
            new_name = f"{entry['source']}_{src_stem}_{country_counters[country]}{src_suffix}"
            dst_path = country_dir / new_name

        shutil.copy2(entry["src_path"], dst_path)

        metadata_rows.append({
            "filename": f"{country}/{new_name}",
            "country": country,
            "continent": entry["continent"],
            "source": entry["source"],
        })

    # Write metadata CSV
    csv_path = OUTPUT_DIR / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "country", "continent", "source"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Print summary
    print(f"\nDone! {len(metadata_rows)} images -> {OUTPUT_DIR}/")
    print(f"Metadata: {csv_path}")

    country_counts = Counter(r["country"] for r in metadata_rows)
    continent_counts = Counter(r["continent"] for r in metadata_rows)
    source_counts = Counter(r["source"] for r in metadata_rows)

    print(f"\n--- By source ---")
    for s, n in source_counts.most_common():
        print(f"  {s:20s} {n:6d}")

    print(f"\n--- By continent ---")
    for c, n in continent_counts.most_common():
        print(f"  {c:20s} {n:6d}")

    print(f"\n--- By country (top 20) ---")
    for c, n in country_counts.most_common(20):
        print(f"  {c:30s} {n:6d}")

    print(f"\n--- By country (bottom 10) ---")
    for c, n in country_counts.most_common()[-10:]:
        print(f"  {c:30s} {n:6d}")

    print(f"\nTotal countries: {len(country_counts)}")
    print(f"Total continents: {len(continent_counts)}")


if __name__ == "__main__":
    main()
