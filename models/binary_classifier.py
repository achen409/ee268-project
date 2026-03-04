import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from poison_dataset import BinaryPoisonDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

# ====================================== data loading =========================================================
clean_dir = "../data_generation/output_data/dog_features"
poison_dir = "../data_generation/output_data/poisoned_dog"
# TODO: windows uses my working dir instead of this dir

# TODO: test varying poisoning ratios (ex. 1%, 5%, 10%)
dataset = BinaryPoisonDataset(clean_dir, poison_dir, poison_ratio=60)

labels = [label for _, label in dataset.samples]

# stratified split: 80 train, 20 test
train_idx, test_idx = train_test_split(
    range(len(dataset.samples)),
    test_size=0.2,
    stratify=labels,
    random_state=67
)

train_dataset = Subset(dataset, train_idx)
test_dataset  = Subset(dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)



# ====================================== model config =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)   # binary
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



# ====================================== training =========================================================
epochs = 5

for epoch in range(epochs):
    model.train()
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


# ====================================== model eval =========================================================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"\nBinary classification accuracy: {accuracy:.2f}%")
print(f"Number of Samples: {total}")
# TODO: confusion matrix
# TODO: more metrics (recall, accuracy, f1-score)