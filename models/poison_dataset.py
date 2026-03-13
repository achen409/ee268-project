import os
import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BinaryPoisonDataset(Dataset):
    def __init__(self, clean_files, poison_files, poison_ratio=0.5, debug=False):
        self.samples = []

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor()
        ])

        if debug:
            print(f"Number of clean files: {len(clean_files)}")
            print(f"Number of poisoned files: {len(poison_files)}")

        num_clean = len(clean_files)

        num_poison = int((poison_ratio / (1 - poison_ratio)) * num_clean)
        num_poison = min(num_poison, len(poison_files))
        
        for f in clean_files:
            self.samples.append((f, 0))

        # poisoned data
        for f in poison_files:
            self.samples.append((f, 1))

        np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with open(path, "rb") as f:
            data = pickle.load(f)

        img = data["img"]

        # numpy to PIL
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = ((img + 1) * 127.5).clip(0,255).astype(np.uint8)
            img = Image.fromarray(img)

        # if already PIL, do nothing
        img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)