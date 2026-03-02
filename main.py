import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, ConcatDataset
import pickle
from PIL import Image
from cifar_image import PoisonedDataset
from cnn_model import SimpleCNN
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import sys

# transform for CIFAR-10
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# download CIFAR-10
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)

# select only car and truck (1, 9)
target_classes = [1, 9]
train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in target_classes]

label_map = {1: 0, 9: 1}
train_dataset = Subset(train_dataset, train_indices)

# map labels inline
train_mapped = []
for img, label in train_dataset:
    label = 0 if label == 1 else 1  # car=0, truck=1
    train_mapped.append((img, torch.tensor(label, dtype=torch.long)))

# adding poison
poisoned_dir = "C:\\Users\\chenk\\Downloads\\nightshade-release-main\\output-data\\poisoned-truck"
poisoned_dataset = PoisonedDataset(poisoned_dir, label=1)

# combine clean CIFAR and poisoned
train_dataset_combined = train_mapped + list(poisoned_dataset)
train_loader = DataLoader(train_dataset_combined, batch_size=32, shuffle=True)


# config
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(target_classes))  # 2 classes: car vs truck
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# training
epochs = 5

for epoch in range(epochs):
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

print("Training complete")



# test on poisoned data
poisoned_dir = "C:\\Users\\chenk\\Downloads\\nightshade-release-main\\output-data\\poisoned-truck"
poisoned_dataset = PoisonedDataset(poisoned_dir)
poisoned_loader = DataLoader(poisoned_dataset, batch_size=16, shuffle=False)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in poisoned_loader:
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

success_rate = 100 * (1 - correct/total)
print(f"Poison attack success rate: {success_rate:.2f}%")