import os
import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BinaryPoisonDataset(Dataset):
    def __init__(self, clean_dir, poison_dir, poison_ratio=0.2, debug=False):
        self.samples = []

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor()
        ])

        # clean data
        clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith(".p")])

        cutoff_index = int(poison_ratio * len(clean_files))
        clean_files = clean_files[cutoff_index:]   # remove duplicates
        if debug:
            print(f"Number of clean files: {len(clean_files)}")

        for f in clean_files:
            self.samples.append((os.path.join(clean_dir, f), 0))

        # poisoned data
        poison_files = sorted([f for f in os.listdir(poison_dir) if f.endswith(".p")])
        poison_files = poison_files[:cutoff_index]
        if debug:
            print(f"Number of poisoned files: {len(poison_files)}")
        for f in poison_files:
            self.samples.append((os.path.join(poison_dir, f), 1))

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