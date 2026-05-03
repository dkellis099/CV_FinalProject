import torch
import torch.nn as nn
import timm
from peft import LoraConfig, get_peft_model

class BiomeCLIPLoRA(nn.Module):
    def __init__(self, num_classes=6):
        super(BiomeCLIPLoRA, self).__init__()
        
        print("Downloading/Loading OpenAI CLIP backbone...")
        self.base_encoder = timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True, num_classes=0)
        self.embed_dim = self.base_encoder.num_features 
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["qkv"],
            lora_dropout=0.05,
            bias="none"
        )
        self.encoder = get_peft_model(self.base_encoder, lora_config)
        
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward_features(self, x):
        """
        Use this method LATER. 
        It returns the raw 768-dimensional CLIP embedding vector.
        """
        return self.encoder(x)

    def forward(self, x):
        """
        Use this method NOW for training.
        """
        features = self.forward_features(x)
        logits = self.head(features)
        return logits