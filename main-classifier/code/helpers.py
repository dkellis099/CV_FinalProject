import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm

_DINOV3_MODEL_CACHE = None
_RESNET_MODEL_CACHE = None
_CLIP_MODEL_CACHE = None
_CONVNEXT_MODEL_CACHE = None


def load_dinov3_encoder(device='cpu'):
    """Load pretrained DINOv3 ViT-Small encoder via timm.

    Returns
    -------
    model : nn.Module
        Frozen DINOv3 encoder. Use model.forward_features(x) to get
        token embeddings, then take token 0 ([CLS]) as the image embedding.
    embed_dim : int
        The embedding dimension (384 for ViT-Small).
    """

    global _DINOV3_MODEL_CACHE
    if _DINOV3_MODEL_CACHE is None:
        print("Downloading DINOv3 model (first time only, ~80 MB)...")
        model = timm.create_model(
            'vit_small_patch16_dinov3_qkvb.lvd1689m',
            pretrained=True,
            num_classes=0,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        _DINOV3_MODEL_CACHE = model

    model = _DINOV3_MODEL_CACHE.to(device)
    _DINOV3_MODEL_CACHE = None
    embed_dim = model.embed_dim  # 384
    return model, embed_dim

def load_resnet50_encoder(device='cpu'):
    """Load pretrained DINOv3 ViT-Small encoder via timm.

    Returns
    -------
    model : nn.Module
        Frozen DINOv3 encoder. Use model.forward_features(x) to get
        token embeddings, then take token 0 ([CLS]) as the image embedding.
    embed_dim : int
        The embedding dimension (384 for ViT-Small).
    """

    global _RESNET_MODEL_CACHE
    if _RESNET_MODEL_CACHE is None:
        print("Downloading ResNet50 model (first time only)...")
        model = timm.create_model(
            'resnet50.a1_in1k',
            pretrained=True,
            num_classes=0,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        _RESNET_MODEL_CACHE = model

    model = _RESNET_MODEL_CACHE.to(device)
    _RESNET_MODEL_CACHE = None
    return model, model.num_features

def load_clip_encoder(device='cpu'):
    """Load pretrained DINOv3 ViT-Small encoder via timm.

    Returns
    -------
    model : nn.Module
        Frozen DINOv3 encoder. Use model.forward_features(x) to get
        token embeddings, then take token 0 ([CLS]) as the image embedding.
    embed_dim : int
        The embedding dimension (384 for ViT-Small).
    """

    global _CLIP_MODEL_CACHE
    if _CLIP_MODEL_CACHE is None:
        print("Downloading Clip model (first time only)...")
        model = timm.create_model(
            "vit_base_patch16_clip_224.laion2b",
            pretrained=True,
            num_classes=0,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        _CLIP_MODEL_CACHE = model

    model = _CLIP_MODEL_CACHE.to(device)
    _CLIP_MODEL_CACHE = None
    return model, model.num_features

def load_convnext_encoder(device='cpu'):
    """Load pretrained DINOv3 ViT-Small encoder via timm.

    Returns
    -------
    model : nn.Module
        Frozen DINOv3 encoder. Use model.forward_features(x) to get
        token embeddings, then take token 0 ([CLS]) as the image embedding.
    embed_dim : int
        The embedding dimension (384 for ViT-Small).
    """

    global _CONVNEXT_MODEL_CACHE
    if _CONVNEXT_MODEL_CACHE is None:
        print("Downloading ConvNext model (first time only)...")
        model = timm.create_model(
            "convnext_tiny.fb_in22k_ft_in1k",
            pretrained=True,
            num_classes=0,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        _CONVNEXT_MODEL_CACHE = model

    model = _CONVNEXT_MODEL_CACHE.to(device)
    _CONVNEXT_MODEL_CACHE = None
    return model, model.num_features