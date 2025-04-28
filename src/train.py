# src/train.py

import time
import torch
import numpy as np
import copy
import os
from torch import optim, nn
from stable_baselines3.common.vec_env import DummyVecEnv

from .models import LatentConceptPredictor, LabelPredictor
from .environment import DynamicConceptEnv
from .utils import compute_fisher_alignment, evaluate_model

def train_full_pipeline_iterative(
    train_features, train_labels,
    val_features, val_labels,
    test_features, test_labels,
    hoc_matrix, hoc_matrix_np,
    num_iterations=5, epochs_per_iter=5,
    batch_size=32, n_steps=64,
    ppo_update_freq=2, significance_threshold=0.1,
    num_concepts=30, num_classes=4, device="cuda"
):
    device = torch.device(device)

    label_predictor = LabelPredictor(num_classes=num_classes, num_concepts=num_concepts).to(device)
    label_optim = optim.Adam(label_predictor.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    all_features = np.concatenate([
        train_features.cpu().numpy(), 
        val_features.cpu().numpy(), 
        test_features.cpu().numpy()
    ], axis=0)
    all_labels = np.concatenate([
        train_labels.cpu().numpy(),
        val_labels.cpu().numpy(),
        test_labels.cpu().numpy()
    ], axis=0)

    alignment_mapping = {}
    iteration_metrics = {}

    latent_predictor = LatentConceptPredictor(
        num_features=2048,
        projected_features=256,
        num_concepts=num_concepts,
        num_envs=batch_size,
        n_steps=n_steps,
        device=device
    )

    for iteration in range(num_iterations):
        print(f"========== Iteration {iteration+1}/{num_iterations} ==========")

        persistent_env = DummyVecEnv([
            lambda i=i: DynamicConceptEnv(
                latent_predictor,
                label_predictor,
                alignment_mapping,
                device,
                entropy_coef=0.1,
                env_index=i
            ) for i in range(batch_size)
        ])
        latent_predictor.ppo_model.set_env(persistent_env)

        best_val_acc = -1.0
        best_model_state = {}

        for epoch in range(epochs_per_iter):
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
            num_batches = 0

            indices = np.random.permutation(len(train_features))
            for i in range(0, len(train_features), batch_size):
                if i + batch_size > len(train_features):
                    continue

                batch_idx = indices[i:i+batch_size]
                batch_features = train_features[batch_idx]
                batch_labels = train_labels[batch_idx]

                persistent_env.env_method("update_batch", batch_features, batch_labels)

                if i % ppo_update_freq == 0:
                    latent_predictor.ppo_model.learn(total_timesteps=n_steps * batch_size, reset_num_timesteps=False)

                actions = latent_predictor.get_action(batch_features)
                actions_binary = (actions > 0.5).astype(int)

                from .utils import translate_batch_latent
                token_batch = translate_batch_latent(actions_binary, alignment_mapping, hoc_matrix_np, num_concepts=num_concepts)

                tokens_tensor = torch.as_tensor(token_batch, dtype=torch.float, device=device)
                preds = label_predictor(tokens_tensor)
                labels_tensor = torch.as_tensor(batch_labels, dtype=torch.long, device=device)

                loss = criterion(preds, labels_tensor)

                label_optim.zero_grad()
                loss.backward()
                label_optim.step()

                with torch.no_grad():
                    batch_acc = (preds.argmax(dim=1) == labels_tensor).float().mean().item()

                epoch_train_loss += loss.item()
                epoch_train_acc += batch_acc
                num_batches += 1

            avg_train_loss = epoch_train_loss / num_batches
            avg_train_acc = epoch_train_acc / num_batches
            val_loss, val_acc = evaluate_model(
                latent_predictor, label_predictor, val_features, val_labels,
                hoc_matrix, alignment_mapping, batch_size, device, criterion
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {
                    'sender_state': latent_predictor.feature_extractor.state_dict(),
                    'receiver_state': label_predictor.state_dict()
                }

        latent_predictor.feature_extractor.load_state_dict(best_model_state['sender_state'])
        label_predictor.load_state_dict(best_model_state['receiver_state'])

        # Test evaluation
        test_loss, test_acc = evaluate_model(
            latent_predictor, label_predictor, test_features, test_labels,
            hoc_matrix, alignment_mapping, batch_size, device, criterion
        )

        print(f"Iteration {iteration+1} - Best Validation Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}")

        iteration_metrics[iteration+1] = {
            "val_acc": best_val_acc,
            "test_acc": test_acc
        }

    return latent_predictor, label_predictor, alignment_mapping, iteration_metrics
