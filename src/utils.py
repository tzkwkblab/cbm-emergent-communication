# src/utils.py

import numpy as np
import torch
from scipy.stats import fisher_exact

def compute_fisher_alignment(latent_matrix, hoc_matrix):
    if isinstance(hoc_matrix, torch.Tensor):
        hoc_matrix = hoc_matrix.cpu().numpy()
    assert latent_matrix.shape[0] == hoc_matrix.shape[0], "Mismatch in sample size."

    num_latent = latent_matrix.shape[1]
    num_hoc = hoc_matrix.shape[1]
    alignment_results = {}

    for i in range(num_latent):
        for j in range(num_hoc):
            a = ((latent_matrix[:, i] == 1) & (hoc_matrix[:, j] == 1)).sum()
            b = ((latent_matrix[:, i] == 1) & (hoc_matrix[:, j] == 0)).sum()
            c = ((latent_matrix[:, i] == 0) & (hoc_matrix[:, j] == 1)).sum()
            d = ((latent_matrix[:, i] == 0) & (hoc_matrix[:, j] == 0)).sum()

            if a + b == 0 or c + d == 0:
                continue
            _, p_value = fisher_exact([[a, b], [c, d]])
            alignment_results[(i, j)] = p_value

    return alignment_results

def compute_alignment_mapping(alignment_results, num_concepts=30, num_hocs=26, threshold=0.1):
    mapping = {}
    for i in range(num_concepts):
        best_p = 1.0
        best_hoc = None
        for j in range(num_hocs):
            p_val = alignment_results.get((i, j), 1.0)
            if p_val < threshold and p_val < best_p:
                best_p = p_val
                best_hoc = j
        if best_hoc is not None:
            mapping[i] = best_hoc
    return mapping

def translate_latent_vector(latent_binary, alignment_mapping, sample_index, hoc_matrix, num_concepts=30):
    token_vector = latent_binary.copy()
    for i in range(num_concepts):
        if latent_binary[i] == 1 and i in alignment_mapping:
            hoc_idx = alignment_mapping[i]
            token_vector[i] = hoc_matrix[sample_index, hoc_idx]
    return token_vector

def translate_batch_latent(latent_binary_batch, alignment_mapping, hoc_matrix, num_concepts=30):
    tokens = []
    batch_size = latent_binary_batch.shape[0]
    for idx in range(batch_size):
        tokens.append(translate_latent_vector(latent_binary_batch[idx], alignment_mapping, idx, hoc_matrix, num_concepts))
    return np.stack(tokens, axis=0)

def evaluate_model(latent_predictor, label_predictor, features, labels, hoc_matrix, alignment_mapping, batch_size, device, criterion):
    label_predictor.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    num_samples = len(features)
    for i in range(0, num_samples, batch_size):
        batch_features = features[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        actions = latent_predictor.get_action(batch_features)
        actions_binary = (actions > 0.5).astype(int)
        token_batch = translate_batch_latent(actions_binary, alignment_mapping, hoc_matrix, num_concepts=latent_predictor.num_concepts)

        tokens_tensor = torch.as_tensor(token_batch, dtype=torch.float, device=device)
        preds = label_predictor(tokens_tensor)
        labels_tensor = torch.as_tensor(batch_labels, dtype=torch.long, device=device)

        loss = criterion(preds, labels_tensor)
        total_loss += loss.item() * len(batch_features)
        total_correct += (preds.argmax(dim=1) == labels_tensor).float().sum().item()
        total_samples += len(batch_features)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc
