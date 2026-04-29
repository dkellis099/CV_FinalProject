import os
import torch
from torchvision import datasets, transforms

# --- CONFIGURATION ---
TRAIN_DIR = './biome_task/raw_data/biome_archive/seg_train'
TEST_DIR = './biome_task/raw_data/biome_archive/seg_test'
OUTPUT_FILE = './biome_task/processed_data/luis_biome_tensors.pt'

resnet_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_directory(directory_path, dataset_name):
    print(f"\nLoading {dataset_name} images from {directory_path}...")
    try:
        dataset = datasets.ImageFolder(root=directory_path, transform=resnet_transforms)
    except FileNotFoundError:
        print(f"Error: Could not find {directory_path}.")
        return []

    print(f"Found {len(dataset)} images in {dataset_name}.")
    
    extracted_data = []
    for i in range(len(dataset)):
        img_tensor, biome_num = dataset[i]
        extracted_data.append((img_tensor, biome_num))
        
        if (i + 1) % 2000 == 0:
            print(f"Processed {i + 1} / {len(dataset)} images...")
            
    return extracted_data

def main():
    processed_data = []
    
    # Process both train and test splits to maximize data for the model
    train_data = process_directory(TRAIN_DIR, "Training Split")
    test_data = process_directory(TEST_DIR, "Testing Split")
    
    processed_data.extend(train_data)
    processed_data.extend(test_data)

    print(f"\nTotal images processed: {len(processed_data)}")
    if len(processed_data) > 0:
        print("Saving master biome dataset to disk...")
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        torch.save(processed_data, OUTPUT_FILE)
        print(f"Success! Biome data ready at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()