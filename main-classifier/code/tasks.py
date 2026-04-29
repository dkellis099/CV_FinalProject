"""
Provided task drivers for HW5 (do not modify).

Task 0: Attention visualization — loads models, calls student.visualize_attention().
Task 4: Transfer evaluation — runs 5 probe experiments, generates attention comparison.
"""

import os
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import encoders as encoders
import hyperparameters as hp
from helpers import create_vit_tiny, load_dinov3_encoder, load_resnet50_encoder, load_clip_encoder, load_convnext_encoder

# ============================================================================
# Task 4: Transfer evaluation
# ============================================================================

frozen_dino = 0
frozen_resnet = 0
frozen_clip = 0
frozen_convnext = 1
last_block_dino = 0
last_block_resnet = 0
last_block_clip = 0
last_block_convnext = 1
lora_dino = 0
lora_clip = 0

def t4_transfer(classify_data, device, approaches, data_dir):
    """Run transfer probes and attention map comparison."""
    num_classes = classify_data.num_classes

    # --- 1. Frozen DINOv3 probe ---
    if frozen_dino:
        print("\n=== Frozen DINOv3 probe ===")
        dinov3_encoder, dinov3_dim = load_dinov3_encoder(device=device)
        model = encoders.GeoEncoder(nn.Linear(dinov3_dim, num_classes),
                                encoder=dinov3_encoder).to(device)
        optimizer = torch.optim.Adam(model.head.parameters(), lr=hp.TRANSFER_HEAD_LR)
        train_accs, val_accs = encoders.train_loop(
            model, classify_data.train_loader, optimizer,
            nn.CrossEntropyLoss(), hp.TRANSFER_EPOCHS, device,
            val_loader=classify_data.val_loader, tasklabel="Frozen-DINOv3")
        np.save(approaches['dinov3_probe'].curve_train, train_accs)
        np.save(approaches['dinov3_probe'].curve_val, val_accs)
        torch.save(model.head.state_dict(), approaches['dinov3_probe'].weights)
    
    # --- 2. Frozen ResNet50 probe ---
    if frozen_resnet:
        print("\n=== Frozen ResNet probe ===")
        resnet50_encoder, resnet50_dim = load_resnet50_encoder(device=device)
        model = encoders.GeoEncoder(nn.Linear(resnet50_dim, num_classes),
                                encoder=resnet50_encoder).to(device)
        optimizer = torch.optim.Adam(model.head.parameters(), lr=hp.TRANSFER_HEAD_LR)
        train_accs, val_accs = encoders.train_loop(
            model, classify_data.train_loader, optimizer,
            nn.CrossEntropyLoss(), hp.TRANSFER_EPOCHS, device,
            val_loader=classify_data.val_loader, tasklabel="Frozen-ResNet50")
        np.save(approaches['resnet50_probe'].curve_train, train_accs)
        np.save(approaches['resnet50_probe'].curve_val, val_accs)
        torch.save(model.head.state_dict(), approaches['resnet50_probe'].weights)
    
    # --- 3. Frozen Clip probe ---
    if frozen_clip:
        print("\n=== Frozen Clip probe ===")
        clip_encoder, clip_dim = load_clip_encoder(device=device)
        model = encoders.GeoEncoder(nn.Linear(clip_dim, num_classes),
                                encoder=clip_encoder).to(device)
        optimizer = torch.optim.Adam(model.head.parameters(), lr=hp.TRANSFER_HEAD_LR)
        train_accs, val_accs = encoders.train_loop(
            model, classify_data.train_loader, optimizer,
            nn.CrossEntropyLoss(), hp.TRANSFER_EPOCHS, device,
            val_loader=classify_data.val_loader, tasklabel="Frozen-Clip")
        np.save(approaches['clip_probe'].curve_train, train_accs)
        np.save(approaches['clip_probe'].curve_val, val_accs)
        torch.save(model.head.state_dict(), approaches['clip_probe'].weights)
    
    # --- 4. Frozen ConvNext probe ---
    if frozen_convnext:
        print("\n=== Frozen ConvNext probe ===")
        convnext_encoder, convnext_dim = load_convnext_encoder(device=device)
        model = encoders.GeoEncoder(nn.Linear(convnext_dim, num_classes),
                                encoder=convnext_encoder).to(device)
        optimizer = torch.optim.Adam(model.head.parameters(), lr=hp.TRANSFER_HEAD_LR)
        train_accs, val_accs = encoders.train_loop(
            model, classify_data.train_loader, optimizer,
            nn.CrossEntropyLoss(), hp.TRANSFER_EPOCHS, device,
            val_loader=classify_data.val_loader, tasklabel="Frozen-ConvNext")
        np.save(approaches['convnext_probe'].curve_train, train_accs)
        np.save(approaches['convnext_probe'].curve_val, val_accs)
        torch.save(model.head.state_dict(), approaches['convnext_probe'].weights)
    
    # --- 5. LoRA DINOv3 probe ---
    if lora_dino:
        print("\n=== LoRA DINOv3 probe ===")
        dinov3_encoder, dinov3_dim = load_dinov3_encoder(device=device)
        model = encoders.GeoEncoder(nn.Linear(dinov3_dim, num_classes),
                                encoder=dinov3_encoder, lora=True).to(device)
        optimizer = torch.optim.Adam([
            {"params": model.head.parameters(), "lr": hp.TRANSFER_HEAD_LR},
            {"params": [p for n, p in model.encoder.named_parameters() if p.requires_grad], "lr": hp.LORA_LR},
        ])
        train_accs, val_accs = encoders.train_loop(
            model, classify_data.train_loader, optimizer,
            nn.CrossEntropyLoss(), hp.TRANSFER_EPOCHS, device,
            val_loader=classify_data.val_loader, tasklabel="LoRA-DINOv3")
        np.save(approaches['dinov3_lora'].curve_train, train_accs)
        np.save(approaches['dinov3_lora'].curve_val, val_accs)
        torch.save(model.head.state_dict(), approaches['dinov3_lora'].weights)
    
    # --- 6. LoRA Clip probe ---
    if lora_clip:
        print("\n=== LoRA Clip probe ===")
        clip_encoder, clip_dim = load_clip_encoder(device=device)
        model = encoders.GeoEncoder(nn.Linear(clip_dim, num_classes),
                                encoder=clip_encoder, lora=True).to(device)
        optimizer = torch.optim.Adam([
            {"params": model.head.parameters(), "lr": hp.TRANSFER_HEAD_LR},
            {"params": [p for n, p in model.encoder.named_parameters() if p.requires_grad], "lr": hp.LORA_LR},
        ])
        train_accs, val_accs = encoders.train_loop(
            model, classify_data.train_loader, optimizer,
            nn.CrossEntropyLoss(), hp.TRANSFER_EPOCHS, device,
            val_loader=classify_data.val_loader, tasklabel="LoRA-Clip")
        np.save(approaches['clip_lora'].curve_train, train_accs)
        np.save(approaches['clip_lora'].curve_val, val_accs)
        torch.save(model.head.state_dict(), approaches['clip_lora'].weights)

    # --- 7. Last Block Dinov3 probe ---
    
    if last_block_dino:
        print("\n=== Last Block DINOv3 probe ===")
        dinov3_encoder, dinov3_dim = load_dinov3_encoder(device=device)
        model = encoders.GeoEncoder(nn.Linear(dinov3_dim, num_classes),
                                encoder=dinov3_encoder).to(device)
        for param in model.encoder.blocks[-1].parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam([
            {"params": model.encoder.blocks[-1].parameters(), "lr": hp.TRANSFER_ENCODER_LR},
            {"params": model.head.parameters(), "lr": hp.TRANSFER_HEAD_LR},
        ])
        train_accs, val_accs = encoders.train_loop(
            model, classify_data.train_loader, optimizer,
            nn.CrossEntropyLoss(), hp.TRANSFER_EPOCHS, device,
            val_loader=classify_data.val_loader, tasklabel="Last-Block-DINOv3")
        np.save(approaches['dinov3_last_block'].curve_train, train_accs)
        np.save(approaches['dinov3_last_block'].curve_val, val_accs)
        torch.save(model.head.state_dict(), approaches['dinov3_last_block'].weights)
    
    # --- 8. Last Block ResNet50 probe ---
    if last_block_resnet:
        print("\n=== Last Block ResNet probe ===")
        resnet50_encoder, resnet50_dim = load_resnet50_encoder(device=device)
        model = encoders.GeoEncoder(nn.Linear(resnet50_dim, num_classes),
                                encoder=resnet50_encoder).to(device)
        for param in model.encoder.layer4.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam([
            {"params": model.encoder.layer4.parameters(), "lr": hp.TRANSFER_ENCODER_LR},
            {"params": model.head.parameters(), "lr": hp.TRANSFER_HEAD_LR},
        ])
        train_accs, val_accs = encoders.train_loop(
            model, classify_data.train_loader, optimizer,
            nn.CrossEntropyLoss(), hp.TRANSFER_EPOCHS, device,
            val_loader=classify_data.val_loader, tasklabel="Last-Block-ResNet50")
        np.save(approaches['resnet50_last_block'].curve_train, train_accs)
        np.save(approaches['resnet50_last_block'].curve_val, val_accs)
        torch.save(model.head.state_dict(), approaches['resnet50_last_block'].weights)
    
    # --- 9. Last Block Clip probe ---
    if last_block_clip:
        print("\n=== Last Block Clip probe ===")
        clip_encoder, clip_dim = load_clip_encoder(device=device)
        model = encoders.GeoEncoder(nn.Linear(clip_dim, num_classes),
                                encoder=clip_encoder).to(device)
        for param in model.encoder.blocks[-1].parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam([
            {"params": model.encoder.blocks[-1].parameters(), "lr": hp.TRANSFER_ENCODER_LR},
            {"params": model.head.parameters(), "lr": hp.TRANSFER_HEAD_LR},
        ])
        train_accs, val_accs = encoders.train_loop(
            model, classify_data.train_loader, optimizer,
            nn.CrossEntropyLoss(), hp.TRANSFER_EPOCHS, device,
            val_loader=classify_data.val_loader, tasklabel="Last-Block-Clip")
        np.save(approaches['clip_last_block'].curve_train, train_accs)
        np.save(approaches['clip_last_block'].curve_val, val_accs)
        torch.save(model.head.state_dict(), approaches['clip_last_block'].weights)
    
    # --- 10. Last Block ConvNext probe ---
    if last_block_convnext:
        print("\n=== Last Block ConvNext probe ===")
        convnext_encoder, convnext_dim = load_convnext_encoder(device=device)
        model = encoders.GeoEncoder(nn.Linear(convnext_dim, num_classes),
                                encoder=convnext_encoder).to(device)
        for param in model.encoder.stages[-1].parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam([
            {"params": model.encoder.stages[-1].parameters(), "lr": hp.TRANSFER_ENCODER_LR},
            {"params": model.head.parameters(), "lr": hp.TRANSFER_HEAD_LR},
        ])
        train_accs, val_accs = encoders.train_loop(
            model, classify_data.train_loader, optimizer,
            nn.CrossEntropyLoss(), hp.TRANSFER_EPOCHS, device,
            val_loader=classify_data.val_loader, tasklabel="Last-Block-ConvNext")
        np.save(approaches['convnext_last_block'].curve_train, train_accs)
        np.save(approaches['convnext_last_block'].curve_val, val_accs)
        torch.save(model.head.state_dict(), approaches['convnext_last_block'].weights)







    # # --- Attention map comparison on high-res test images ---
    # print("\n=== Attention map comparison ===")
    # to_tensor = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # img_dir = os.path.join(data_dir, 'highres-images')
    # sample_paths = sorted([
    #     os.path.join(img_dir, f) for f in os.listdir(img_dir)
    #     if f.endswith(('.jpg', '.png'))
    # ]) if os.path.isdir(img_dir) else []

    # # Load all encoders for comparison
    # random_enc, _ = create_vit_tiny()
    # random_enc.eval()
    # rotation_enc, _ = create_vit_tiny()
    # rotation_enc.load_state_dict(
    #     torch.load(approaches['rotation'].weights, weights_only=True, map_location='cpu'))
    # rotation_enc.eval()
    # dino_enc, _ = create_vit_tiny()
    # dino_enc.load_state_dict(
    #     torch.load(approaches['dino'].weights, weights_only=True, map_location='cpu'))
    # dino_enc.eval()

    # for img_path in sample_paths[:3]:
    #     img_name = os.path.splitext(os.path.basename(img_path))[0]
    #     tensor = to_tensor(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)

    #     for label, enc in [('random', random_enc), ('rotation', rotation_enc),
    #                        ('dino', dino_enc), ('dinov3', dinov3_encoder)]:
    #         student.visualize_attention(
    #             enc, tensor, os.path.join('results', f'compare_{img_name}_fade_{label}.png'),
    #             style='fade', device=device)
    #         student.visualize_attention(
    #             enc, tensor, os.path.join('results', f'compare_{img_name}_gray_{label}.png'),
    #             style='gray', device=device)
