# main.py

import argparse
import torch
import numpy as np
import pandas as pd

from src.config import *
from src.train import train_full_pipeline_iterative

def set_random_seed(seed):
    import random
    from stable_baselines3.common.utils import set_random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_random_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features", default=TRAIN_FEATURES_PATH)
    parser.add_argument("--val_features", default=VAL_FEATURES_PATH)
    parser.add_argument("--test_features", default=TEST_FEATURES_PATH)
    parser.add_argument("--hoc_csv", default=HOC_CSV_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(SEED)

    train_data = np.load(args.train_features)
    val_data = np.load(args.val_features)
    test_data = np.load(args.test_features)

    train_features = torch.tensor(train_data["features"], dtype=torch.float32, device=device)
    train_labels = torch.tensor(train_data["labels"], dtype=torch.long, device=device)
    val_features = torch.tensor(val_data["features"], dtype=torch.float32, device=device)
    val_labels = torch.tensor(val_data["labels"], dtype=torch.long, device=device)
    test_features = torch.tensor(test_data["features"], dtype=torch.float32, device=device)
    test_labels = torch.tensor(test_data["labels"], dtype=torch.long, device=device)

    hoc_df = pd.read_csv(args.hoc_csv)
    hoc_df = hoc_df.dropna(axis=0)
    hoc_matrix = torch.tensor(hoc_df.iloc[:, 1:].to_numpy(dtype=int), device=device)
    hoc_matrix_np = hoc_matrix.cpu().numpy()

    latent_predictor, label_predictor, alignment_mapping, iteration_metrics = train_full_pipeline_iterative(
        train_features, train_labels,
        val_features, val_labels,
        test_features, test_labels,
        hoc_matrix, hoc_matrix_np,
        num_iterations=NUM_ITERATIONS,
        epochs_per_iter=EPOCHS_PER_ITER,
        batch_size=BATCH_SIZE,
        n_steps=N_STEPS,
        ppo_update_freq=PPO_UPDATE_FREQ,
        significance_threshold=SIGNIFICANCE_THRESHOLD,
        num_concepts=NUM_CONCEPTS,
        num_classes=NUM_CLASSES,
        device=device
    )

if __name__ == "__main__":
    main()
