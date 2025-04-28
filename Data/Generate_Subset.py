import os
import re
import argparse
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import OxfordIIITPet

# -------------------------------
# Dataset Class
# -------------------------------
class CatBreedDataset(Dataset):
    def __init__(self, transformed_images, filtered_breed_dict):
        self.data = []
        breeds = sorted(filtered_breed_dict.keys())
        for breed in breeds:
            label = breeds.index(breed)
            for filename, image_tensor in transformed_images[breed]:
                self.data.append((image_tensor, label, filename))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# -------------------------------
# Main Function
# -------------------------------
def main(hoc_csv_path, output_dir):
    # Download Oxford Pet dataset
    trainval_dataset = OxfordIIITPet(root='./data', split='trainval', download=True)
    test_dataset = OxfordIIITPet(root='./data', split='test', download=True)
    all_images = trainval_dataset._images + test_dataset._images
    all_labels = list(trainval_dataset._labels) + list(test_dataset._labels)

    # Build full mapping: breed → [(filename, full_path, label)]
    breed_dict = defaultdict(list)
    for filepath, label in zip(all_images, all_labels):
        breed = trainval_dataset.classes[label].replace(" ", "_")
        filename = os.path.splitext(os.path.basename(filepath))[0]
        breed_dict[breed].append((filename, filepath, label))

    # Load HOC file
    hoc_data = pd.read_csv(hoc_csv_path)
    image_names = hoc_data.iloc[:, 0].dropna().tolist()

    # Filter only annotated images
    filtered_images = [entry for breed in breed_dict for entry in breed_dict[breed] if entry[0] in image_names]
    filtered_breed_dict = defaultdict(list)
    for filename, filepath, label in filtered_images:
        breed = trainval_dataset.classes[label].replace(" ", "_")
        filtered_breed_dict[breed].append(filename)

    # Sort numerically
    def extract_number(filename):
        match = re.search(r'_(\d+)', filename)
        return int(match.group(1)) if match else 0

    for breed in filtered_breed_dict:
        filtered_breed_dict[breed].sort(key=extract_number)

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transformed_images = {}
    for breed in filtered_breed_dict:
        transformed_images[breed] = []
        for filename in filtered_breed_dict[breed]:
            image_path = next((fp for fn, fp, _ in breed_dict[breed] if fn == filename), None)
            if image_path and os.path.exists(image_path):
                img = Image.open(image_path).convert("RGB")
                img_transformed = transform(img)
                transformed_images[breed].append((filename, img_transformed))

    dataset = CatBreedDataset(transformed_images, filtered_breed_dict)
    all_data = dataset.data
    labels = [sample[1] for sample in all_data]

    train_data, temp_data = train_test_split(all_data, test_size=0.2, stratify=labels, random_state=42)
    temp_labels = [sample[1] for sample in temp_data]
    val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_labels, random_state=42)

    # Save splits
    os.makedirs(output_dir, exist_ok=True)

    def save_split(data_split, filename):
        images = np.stack([img.numpy() for img, _, _ in data_split])
        labels = np.array([label for _, label, _ in data_split])
        fnames = np.array([fname for _, _, fname in data_split])
        np.savez(os.path.join(output_dir, filename), images=images, labels=labels, filenames=fnames)
        print(f"Saved {filename}")

    save_split(train_data, "train.npz")
    save_split(val_data, "val.npz")
    save_split(test_data, "test.npz")

# -------------------------------
# CLI Entry Point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hoc_csv", required=True, help="Path to your HOC annotation CSV file")
    parser.add_argument("--output_dir", default="./processed_data", help="Where to save the .npz files")
    args = parser.parse_args()

    main(args.hoc_csv, args.output_dir)