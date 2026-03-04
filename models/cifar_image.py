import os
import pickle
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class PoisonedDataset(Dataset):
    def __init__(self, data_dir, label=1, transform=None):
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pkl")])
        self.label = label
        self.transform = transform if transform is not None else T.Compose([
            T.Resize((128, 128)),   # resize for CNN
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as f:
            data = pickle.load(f)
        img = data["img"]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.label, dtype=torch.long)