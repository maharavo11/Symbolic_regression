# Central configuration file for hyperparameters and settings.

import torch
import sympy as sp
import random
import numpy as np

# --- Random Seeds ---
RANDOM_SEED = 123
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# --- Operators and Leaves ---
BINARY_OPERATORS = {
    "+": lambda l, r: l + r,
    "-": lambda l, r: l - r,
    "*": lambda l, r: l * r,
}
UNARY_OPERATORS = {
    "abs": lambda o: sp.Abs(o),
    "sin": lambda o: sp.sin(o),
    "tan": lambda o: sp.tan(o),
    "exp": lambda o: sp.exp(o),
}
LEAVES = ("x",)

# --- Data Generation ---
NUM_EXPRESSIONS_TO_GENERATE = 50000  # Initial number before filtering
MAX_BINARY_OPS = 4
MAX_UNARY_OPS = 3
NUM_DATA_POINTS = 100  # Number of (x, f(x)) points per expression
X_RANGE_MIN = -1
X_RANGE_MAX = 1
FILTER_Y_MAX_ABS_VALUE = 10

# --- Model Hyperparameters (SeqGRU) ---
D_MODEL = 128
DEC_HIDDEN_DIM = 128
N_GRU_LAYERS = 2
N_ENC_LAYERS = 2

# --- Training Hyperparameters ---
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_TRAINING_STEPS = 15000

# --- Evaluation & Plotting ---
NUM_PREDICTION_PLOTS = 12
OUTPUT_DIR = "outputs"

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")