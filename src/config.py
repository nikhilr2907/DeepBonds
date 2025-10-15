"""Configuration and Constants for DeepBonds"""

import random
import numpy as np
import torch

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Model hyperparameters
SEQUENCE_LENGTH = 22
INPUT_SIZE = 3
HIDDEN_SIZE = 50
NUM_LAYERS = 2
BATCH_SIZE = 50
LEARNING_RATE = 0.006
N_EPOCHS = 500
TARGET_LENGTH = 22

# Data split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.75  # 75% of training data for validation split

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
