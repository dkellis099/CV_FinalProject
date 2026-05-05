"""
Hyperparameters for HW5: Vision Transformers and Self-Supervised Learning.

You may modify these values to improve your results.
"""

# ============================================================================
# Task 4: Transfer Evaluation
# ============================================================================
TRANSFER_EPOCHS = 15
TRANSFER_HEAD_LR = 1e-3           # Learning rate for linear head
TRANSFER_ENCODER_LR = 1e-5        # Learning rate for encoder (finetuning only)
TRANSFER_WEIGHT_DECAY = 0.01
LORA_LR = 1e-4
BATCH_SIZE = 32
IMAGE_SIZE = 224
