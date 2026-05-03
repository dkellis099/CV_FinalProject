import os

TRAIN_DATA_PATH = '../biome_task/processed_data/train_biome_tensors.pt'
TEST_DATA_PATH = '../biome_task/processed_data/test_biome_tensors.pt'
SAVE_DIR = './checkpoints'

NUM_CLASSES = 6

BATCH_SIZE = 8 
LEARNING_RATE = 3e-4 
EPOCHS = 5

os.makedirs(SAVE_DIR, exist_ok=True)