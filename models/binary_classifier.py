import os
import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from poison_dataset import BinaryPoisonDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Subset, DataLoader

# TODO: move model training & eval to function(s)
# TODO: training -> params: train_loader, test_loader, debug=False, poison_ratio, num_epochs. return model, losses
# TODO: test -> params: model, debug=False. return report_dict, cm
# ====================================== data loading =========================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

# absolute paths relative to the script
clean_dir  = os.path.join(script_dir, "..", "data_generation", "output_data", "dog_features")
poison_dir = os.path.join(script_dir, "..", "data_generation", "output_data", "poisoned_dog")

# TODO: test varying poisoning ratios (ex. 1%, 5%, 10%)
dataset = BinaryPoisonDataset(clean_dir, poison_dir, poison_ratio=10)

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
model.fc = nn.Linear(model.fc.in_features, 2) # binary
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



# ====================================== training =========================================================
epochs = 15
losses = []
# TODO: more epochs since it works so fast? varying epochs?

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
    losses.append(running_loss/len(train_loader))

print("Training complete")


# ====================================== model eval =========================================================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)
# TN FP
# FN TP

# more metrics
# precision: TP / (TP + FP)
# recall: TP / (TP + FN)
report_dict = classification_report(all_labels, all_preds, target_names=["Clean", "Poison"], output_dict=True)
report_str = classification_report(all_labels, all_preds, target_names=["Clean", "Poison"])
print("\nClassification Report:")
print(report_str)