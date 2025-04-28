# src/config.py

# Dataset paths ( Adjust if needed )
TRAIN_FEATURES_PATH = "./features/subset_train_features.npz"
VAL_FEATURES_PATH = "./features/subset_val_features.npz"
TEST_FEATURES_PATH = "./features/subset_test_features.npz"
HOC_CSV_PATH = "./processed_data/hoc_annotations.csv" 

# Model parameters
NUM_CONCEPTS = 30
NUM_CLASSES = 4
PROJECTED_FEATURES_DIM = 256
NUM_HOCS = 26  # Adjust if your HOC file has a different number of attributes

# Training parameters
BATCH_SIZE = 32
N_STEPS = 64
NUM_ITERATIONS = 5
EPOCHS_PER_ITER = 30
PPO_UPDATE_FREQ = 64
SIGNIFICANCE_THRESHOLD = 0.1

# Random seed
SEED = 42
