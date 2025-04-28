import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

# -------------------------------
# Custom Dataset from .npz
# -------------------------------
class NPZImageDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.images = torch.tensor(data["images"])
        self.labels = torch.tensor(data["labels"])
        self.filenames = data["filenames"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.filenames[idx]

# -------------------------------
# Feature Extraction
# -------------------------------
def extract_features_and_save(model, loader, save_path, device):
    features_list, labels_list, filenames_list = [], [], []

    with torch.no_grad():
        for images, batch_labels, batch_fnames in tqdm(loader, desc=f"Extracting features: {save_path}"):
            images = images.to(device)
            batch_features = model(images).flatten(1).cpu().numpy()
            features_list.append(batch_features)
            labels_list.extend(batch_labels.numpy())
            filenames_list.extend(batch_fnames)

    features = np.concatenate(features_list, axis=0)
    labels = np.array(labels_list)
    fnames = np.array(filenames_list)
    np.savez(save_path, features=features, labels=labels, filenames=fnames)
    print(f"✅ Features saved to: {save_path}")

# -------------------------------
# Main
# -------------------------------
def main(train_npz, val_npz, test_npz, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet50 = models.resnet50(pretrained=True)
    resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])
    resnet50.to(device).eval()

    transform = lambda x: x  # No transform needed if already preprocessed

    for split, npz_path in zip(["train", "val", "test"], [train_npz, val_npz, test_npz]):
        dataset = NPZImageDataset(npz_path)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        output_path = os.path.join(output_dir, f"subset_{split}_features.npz")
        extract_features_and_save(resnet50, loader, output_path, device)

# -------------------------------
# CLI Entry Point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", required=True, help="Path to train.npz")
    parser.add_argument("--val_npz", required=True, help="Path to val.npz")
    parser.add_argument("--test_npz", required=True, help="Path to test.npz")
    parser.add_argument("--output_dir", default="./features", help="Where to save feature .npz files")
    args = parser.parse_args()

    main(args.train_npz, args.val_npz, args.test_npz, args.output_dir)