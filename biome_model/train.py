import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

import config as cfg
from dataset import BiomeDataset
from model import BiomeCLIPLoRA

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Initializing training on {device}...")

    train_dataset = BiomeDataset(cfg.TRAIN_DATA_PATH)
    val_dataset = BiomeDataset(cfg.TEST_DATA_PATH)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    model = BiomeCLIPLoRA(num_classes=cfg.NUM_CLASSES).to(device)
    
    model.encoder.print_trainable_parameters()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)

    print("\n--- Starting CLIP+LoRA Training ---")
    best_val_acc = 0.0
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{cfg.EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"-> Epoch {epoch+1} Summary | Train Loss: {running_loss/len(train_loader):.4f} | Val Accuracy: {val_acc:.2f}%\n")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(cfg.SAVE_DIR, 'best_biome_clip_lora.pth')
            torch.save(model.state_dict(), save_path)
            print(f"*** New best model saved to {save_path} ***\n")

    print("Training complete!")

if __name__ == "__main__":
    main()