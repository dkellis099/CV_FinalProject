import torch
from torchvision import transforms
from PIL import Image

import config as cfg
from model import BiomeCLIPLoRA 

device = torch.device("cpu") 

print("Loading Biome Model...")
biome_model = BiomeCLIPLoRA(num_classes=6).to(device)
biome_model.load_state_dict(torch.load("best_biome_clip_lora.pth", map_location=device), strict=False)
biome_model.eval()

luis_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# wrapper function
def get_biome_logits(image):
    """
    Wrapper function for Sam and Dylan.
    Input: A raw PIL Image object.
    Output: A PyTorch tensor containing 6 raw logits.
    """
    img_tensor = luis_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = biome_model(img_tensor)[0]
        
    return logits