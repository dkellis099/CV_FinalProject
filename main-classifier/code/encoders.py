import os
import copy
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model
from pathlib import Path
from helpers import load_clip_encoder

import hyperparameters as hp
from helpers import load_clip_encoder
from biome_model import BiomeCLIPLoRA

SEED = 1865415
torch.manual_seed(SEED)

# ========================================================================
# GeoEncoder - Generalized encoder for GeoGuessr model
# ========================================================================
#
class GeoEncoder(nn.Module):
    """ViT backbone with a head on the [class] token.

    Used for classification (Linear head), rotation (Linear head),
    DINO (MLP head), and DINOv3 (by passing encoder= to constructor).

    Arguments:
        head    -- nn.Module to apply to the [class] token embedding
        encoder -- optional external encoder (default: creates ViT-Tiny)

    After construction, provides:
        .encoder     -- the ViT backbone
        .encoder_dim -- embedding dimension (192 for ViT-Tiny, 384 for DINOv3)
        .head        -- the head module
    """

    def __init__(self, encoder=None, lora=False, freeze_last_layer=False, device="cpu", num_classes=6):
        super().__init__()
        self.encoder = encoder
        if lora:
            self.lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["qkv"],
                lora_dropout=0.05,
                bias="none",
            )
        self.encoder = get_peft_model(self.encoder, self.lora_config)
        self.device = device
        self.biome_model = BiomeCLIPLoRA(num_classes=6).to(self.device)
        state = torch.load(Path(__file__).resolve().parents[2] / "model_weights/best_biome_clip_lora.pth", map_location=self.device)
        self.biome_model.load_state_dict(state)
        self.biome_model.eval()
    
        self.sun_encoder, self.sun_dim = load_clip_encoder(device=device)
        lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["qkv"], lora_dropout=0.05, bias="none")
        self.sun_encoder = get_peft_model(self.sun_encoder, lora_config)

        state = torch.load(Path(__file__).resolve().parents[2] / "model_weights/clip_lora_encoder.pt", map_location=self.device)
        self.sun_encoder.load_state_dict(state)
        self.sun_encoder.to(device)
        self.sun_encoder.eval()
        self.head = nn.Linear(self.encoder.num_features + self.biome_model.embed_dim + self.sun_dim, num_classes)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.num_features + self.biome_model.embed_dim + self.sun_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        tokens = self.encoder.forward_features(x)
        tokens = tokens[:, 0, :] 
        # print(tokens)

        for p in self.biome_model.parameters():
            p.requires_grad = False
        for p in self.sun_encoder.parameters():
            p.requires_grad = False

        with torch.no_grad():
            biome_features = self.biome_model.forward_features(x)
            sun_features = self.sun_encoder(x)
            # print(biome_features)
        
        # print(tokens.shape)
        # print(biome_features.shape)
        tokens = torch.cat([F.normalize(tokens, dim=1), F.normalize(biome_features, dim=1), F.normalize(sun_features, dim=1)], dim=1)
        
        # if tokens.dim() == 4:
        #     cls_token = tokens.mean(dim=(2, 3))
        # else:
        #     cls_token = tokens[:, 0, :]

        return self.head(tokens)