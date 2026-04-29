import os
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import reverse_geocoder as rg
import pycountry_convert as pc

# --- CONFIGURATION ---
DATA_SPLITS = [
    {
        'name': 'Train Split',
        'csv': './continent_task/raw_data/continent_archive/train/metadata.csv',
        'img_dir': './continent_task/raw_data/continent_archive/train/images'
    },
    {
        'name': 'Test Split',
        'csv': './continent_task/raw_data/continent_archive/test/metadata.csv',
        'img_dir': './continent_task/raw_data/continent_archive/test/images'
    }
]

OUTPUT_FILE = './continent_task/processed_data/luis_continent_tensors.pt'

continent_mapping = {
    'NA': 1, 'SA': 2, 'AF': 3, 'EU': 4, 'AS': 5, 'OC': 6
}

resnet_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_continent_from_coords(lat, lon):
    try:
        results = rg.search((lat, lon), mode=1)
        country_code = results[0]['cc']
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        return continent_mapping.get(continent_code, None)
    except Exception:
        return None

def main():
    processed_dataset = []
    total_success, total_errors = 0, 0
    
    # Iterate over both the train and test splits
    for split in DATA_SPLITS:
        print(f"\n--- Processing {split['name']} ---")
        try:
            df = pd.read_csv(split['csv'])
        except FileNotFoundError:
            print(f"Error: Could not find CSV at {split['csv']}")
            continue

        print(f"Found {len(df)} rows in {split['name']} metadata.")
        
        for index, row in df.iterrows():
            try:
                # Using the exact fields defined in the PM25Vision README
                img_name = row['filename'] 
                lat = float(row['latitude'])
                lon = float(row['longitude'])
            except KeyError as e:
                print(f"Column error: {e}. Row skipped.")
                total_errors += 1
                continue
                
            img_path = os.path.join(split['img_dir'], img_name)
            continent_num = get_continent_from_coords(lat, lon)
            
            if not continent_num or not os.path.exists(img_path):
                total_errors += 1
                continue
                
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = resnet_transforms(img)
                
                processed_dataset.append((img_tensor, continent_num))
                total_success += 1
                
                if total_success % 1000 == 0:
                    print(f"Successfully mapped and processed {total_success} total images...")
                    
            except Exception:
                total_errors += 1

    print(f"\n--- Final Results ---")
    print(f"Success: {total_success} images converted to tensors.")
    print(f"Skipped/Errors: {total_errors} (Usually coordinates over oceans or missing files)")
    
    if total_success > 0:
        print("Saving master continent dataset to disk...")
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        torch.save(processed_dataset, OUTPUT_FILE)
        print(f"Success! Continent data ready at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()