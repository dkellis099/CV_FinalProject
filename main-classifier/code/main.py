import os
import argparse
from collections import namedtuple

import torch

import encoders as encoders
import utils as utils
import hyperparameters as hp
from tasks import geoguessr

# ============================================================================
# Output file tracking
# ============================================================================

Approach = namedtuple('Approach', ['label', 'weights', 'curve_train', 'curve_val'])

os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)

APPROACHES = {
    'dinov3_probe':   Approach('Frozen DINOv3 probe',             'results/dinov3_probe.pt',         'results/train_dinov3_probe.npy',   'results/val_dinov3_probe.npy'),
    'resnet50_probe': Approach('Frozen ResNet50 probe',           'results/resnet50_probe.pt',       'results/train_resnet50_probe.npy', 'results/val_resnet50_probe.npy'),
    'clip_probe':     Approach('Frozen Clip probe',           'results/clip_probe.pt',       'results/train_clip_probe.npy', 'results/val_clip_probe.npy'),
    'convnext_probe': Approach('Frozen ConvNext probe',           'results/convnext_probe.pt',       'results/train_convnext_probe.npy', 'results/val_convnext_probe.npy'),
    'dinov3_lora': Approach('LoRA DINOv3 probe',           'results/dinov3_lora.pt',       'results/train_dinov3_lora.npy', 'results/val_dinov3_lora.npy'),
    'clip_lora':     Approach('LoRA Clip probe',           'results/clip_lora.pt',       'results/train_clip_lora.npy', 'results/val_clip_lora.npy'),
    'dinov3_last_block':   Approach('Last Block DINOv3 probe',             'results/dinov3_last_block.pt',         'results/train_dinov3_last_block.npy',   'results/val_dinov3_last_block.npy'),
    'resnet50_last_block': Approach('Last Block ResNet50 probe',           'results/resnet50_last_block.pt',       'results/train_resnet50_last_block.npy', 'results/val_resnet50_last_block.npy'),
    'clip_last_block':     Approach('Last Block Clip probe',           'results/clip_last_block.pt',       'results/train_clip_last_block.npy', 'results/val_clip_last_block.npy'),
    'convnext_last_block': Approach('Last Block ConvNext probe',           'results/convnext_last_block.pt',       'results/train_convnext_last_block.npy', 'results/val_convnext_last_block.npy'),
}


# ============================================================================
# Task dispatchers
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Final Project: Geoguessr Model')
    parser.add_argument('--task', type=str, required=True,
                        choices=['geoguessr'],
                        help='Which task to run.')
    parser.add_argument('--data', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data'),
                        help='Path to data directory.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Task: {args.task}")
    print(f"SEED: {encoders.SEED}")
    torch.manual_seed(encoders.SEED)

    if args.task == 'geoguessr':
        classify_data = utils.SceneDataset(
            os.path.join(os.path.dirname(__file__), '50k_data'),
            image_size=hp.IMAGE_SIZE,
            batch_size=hp.BATCH_SIZE,
        )
        geoguessr(classify_data, device, APPROACHES, args.data)

    print(f"\nTask {args.task} complete.")


if __name__ == '__main__':
    main()
