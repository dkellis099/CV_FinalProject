"""
Homework 5 - Vision Transformers and Self-Supervised Learning
CSCI1430 - Computer Vision
Brown University
"""

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

import hyperparameters as hp
from helpers import create_vit_tiny, get_attention_weights, DINODashboard
from utils import train_loop, SceneDataset, CropRotationDataset

BANNER_ID = 1865415 # <- replace with your Banner ID; drop the 'B' prefix and any leading 0s.
torch.manual_seed(BANNER_ID)

# ========================================================================
#  TASK 0: Attention map visualization
#
#  Visualize what ViT attention heads "look at."
#  We extract [class]-to-patch attention from the last transformer layer
#  and display it in two styles: fade-to-black and grayscale heatmaps.
#  This function is reused throughout the homework (Tasks 0, 3, 4).
# ========================================================================

# Part A: Visualize attention maps
#
def visualize_attention(model, image_tensor, save_path, style='fade', device='cpu'):
    """Extract and visualize [class]-to-patch attention from a ViT.

    This function does two things:
      1. Extract attention: call get_attention_weights() to get the raw
         (num_heads, num_tokens, num_tokens) matrix from the last layer,
         then pull out [class]'s attention to each patch and reshape to 2D.
      2. Visualize: display the original image alongside per-head attention maps.

    Two visualization styles:
        'gray'  -- Nearest-neighbor upsample (preserves pixelated patch grid),
                   display as grayscale.  (Caron et al. DINO style)
        'fade'  -- Bilinear upsample to image resolution, multiply image by
                   attention. High-attention areas stay visible; low-attention
                   areas fade to black.  (Dosovitskiy et al. style)

    Rescaling for display: 
    Each head's [class]-to-patch attention is a probability distribution (softmax), 
    so the values sum to 1 across all patches. But, suppose we have a 
    14x14 = 196 patch grid, then a uniform attention gives ~0.005 per patch — 
    nearly black. Even a head that focuses on a single region might peak at 0.1. 
    
    To make the patterns visible, we can rescale each head's attention to [0, 1] 
    via (a - a.min()) / (a.max() - a.min()). This stretches whatever variation 
    exists to fill the display range. Note that this means a nearly-uniform head 
    can look structured — compare the range (max - min) across heads to judge 
    which are truly selective.

    Arguments:
        model        -- a timm ViT model (e.g., from create_vit_tiny())
        image_tensor -- (1, 3, H, W) tensor, values in [0, 1]
        save_path    -- where to save the PNG
        style        -- 'fade' or 'gray'
        device       -- torch device

    Hints:
        - get_attention_weights(model, image_tensor, device) returns shape
          (num_heads, num_tokens, num_tokens). Token 0 is [class].
        - model.num_prefix_tokens tells you how many non-patch tokens (usually 1).
        - For a 224px image with 16px patches: 14x14 = 196 patches.
        - Use F.interpolate to upsample, mode='bilinear' or mode='nearest'.
    """
    # TODO:
    #   Step 1: Extract attention maps
    #   1. Call get_attention_weights(model, image_tensor, device)
    #      -> (num_heads, num_tokens, num_tokens)
    normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    
    image_for_display = image_tensor.clone()
    image_for_model = normalize(image_tensor.clone())

    attention_weights = get_attention_weights(model=model, image_tensor=image_for_model, device=device)

    #   2. Extract [class] row: attn[:, 0, num_prefix:]
    #      -> (num_heads, num_patches)
    class_weights = attention_weights[:, 0, model.num_prefix_tokens:]

    #   3. Reshape to 2D grid. The grid shape depends on the image size:
    H_img, W_img = image_tensor.shape[2], image_tensor.shape[3]
    num_heads = class_weights.shape[0]
    patch_size = 16
    h, w = H_img // patch_size, W_img // patch_size
    class_weights = class_weights.reshape(num_heads, h, w)

    #
    #   Step 2: Build visualization panels
    #
    #   Convert image_tensor[0] to numpy (H, W, 3) for display.
    #
    #   Implement style='gray' first — simpler:
    #   4. For each head:
    #      a. Rescale attention to [0, 1] (see note above).
    #      b. Nearest-neighbor upsample: F.interpolate(..., mode='nearest')
    #      -> produces a list of upsampled attention maps (as numpy arrays)
    attn_maps = []
    for head in range(num_heads):
        a = class_weights[head]
        a = (a - a.min()) / (a.max() - a.min())
        attn_maps.append(a)

    attn_maps = torch.stack(attn_maps, dim=0).unsqueeze(1)

    if style == 'gray':
        attn_up = F.interpolate(attn_maps, size=(H_img, W_img), mode='nearest')
    else:
        attn_up = F.interpolate(attn_maps, size=(H_img, W_img), mode='bilinear', align_corners=False)

    #   Then add style='fade':
    #   4. For each head:
    #      a. Rescale attention to [0, 1] (see note above).
    #      b. Bilinear upsample: F.interpolate(..., mode='bilinear')
    #      c. Fade the input image = image * attn_up[:, :, np.newaxis]
    #      -> produces a list of faded images (as numpy arrays)
    attn_up = attn_up.squeeze(1).numpy()
    image_np = image_for_display[0].cpu().permute(1, 2, 0).numpy()

    #
    #   Step 3: Make figure and save
    #   5. fig, axes = plt.subplots(1, num_heads + 1, ...)
    #      First panel: original image. Remaining panels: attention maps.
    #   6. imshow each panel. For 'gray': cmap='gray', vmin=0, vmax=1.
    #   7. fig.savefig(save_path, ...) then plt.close(fig)
    fig, axes = plt.subplots(1, num_heads + 1, figsize=(3 * (num_heads + 1), 3))

    axes[0].imshow(image_np)
    axes[0].set_title("original")
    axes[0].axis("off")

    for head in range(num_heads):
        if style == 'gray':
            axes[head + 1].imshow(attn_up[head], cmap='gray', vmin=0, vmax=1)
        else:
            faded = image_np * attn_up[head][:, :, np.newaxis]
            axes[head + 1].imshow(faded)

        axes[head + 1].set_title(f"head {head}")
        axes[head + 1].axis("off")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
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

    def __init__(self, head, encoder=None, lora=False, freeze_last_layer=False):
        super().__init__()
        self.encoder = encoder
        self.head = head
        if lora:
            self.lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["qkv"],
                lora_dropout=0.05,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, self.lora_config)

    def forward(self, x):
        tokens = self.encoder.forward_features(x)
        if tokens.dim() == 4:
            cls_token = tokens.mean(dim=(2, 3))
        else:
            cls_token = tokens[:, 0, :]

        return self.head(cls_token)