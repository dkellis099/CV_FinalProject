import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

import config as cfg
from model import BiomeCLIPLoRA

def predict_image(image_path, weights_path):
    device = torch.device("cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    print("Building model architecture...")
    model = BiomeCLIPLoRA(num_classes=cfg.NUM_CLASSES).to(device)
    
    print(f"Loading learned weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    
    model.eval() 

    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device) 
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{image_path}'")
        return

    print(f"\nAnalyzing '{image_path}'...\n")
    with torch.no_grad():
        outputs = model(img_tensor)
        
        # --- THE DIAGNOSTIC PRINT ---
        print("RAW LOGITS (The pure math):")
        print(outputs[0])
        print("--------------------------\n")
        
        probabilities = F.softmax(outputs[0], dim=0)
        
    print("--- PREDICTION RESULTS ---")
    for i, prob in enumerate(probabilities):
        print(f"{classes[i]:>10}: {prob.item() * 100:.2f}%")
        
    best_guess_idx = torch.argmax(probabilities).item()
    print(f"\n🏆 WINNER: {classes[best_guess_idx].upper()}")

if __name__ == "__main__":
    MY_TEST_IMAGE = "test_image.jpg" 
    MY_WEIGHTS = "./checkpoints/best_biome_clip_lora.pth"
    
    predict_image(MY_TEST_IMAGE, MY_WEIGHTS)