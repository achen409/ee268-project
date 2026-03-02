from cifar_image import PoisonedDataset
from cnn_model import SimpleCNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# config
data_dir = "C:/Users/chenk/Downloads/nightshade-release-main/output-data/poisoned-truck"
batch_size = 8
epochs = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

# dataset and loader
dataset = PoisonedDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# model, loss, optimizer
model = SimpleCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# dummy labels
labels = torch.tensor([i % 2 for i in range(len(dataset))])

# training loop
for epoch in range(epochs):
    running_loss = 0.0
    for i, (imgs, texts) in enumerate(dataloader):
        imgs = imgs.to(device)
        # Use dummy labels for now
        batch_labels = labels[i*batch_size:i*batch_size+imgs.size(0)].to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

print("Training done!")