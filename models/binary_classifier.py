import os
import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from poison_dataset import BinaryPoisonDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

class BinaryPoisonClassifier():
    # initialization
    def __init__(self, poison_ratio=0.3, epochs=15, batch_size=32):
        self.poison_ratio = poison_ratio
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.load_data()
        self.init_model()

    # ====================================== data loading =========================================================
    def load_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # absolute paths relative to the script
        clean_dir  = os.path.join(script_dir, "..", "data_generation", "output_data", "dog_features")
        poison_dir = os.path.join(script_dir, "..", "data_generation", "output_data", "poisoned_dog")

        clean_files = [os.path.join(clean_dir,f) for f in os.listdir(clean_dir) if f.endswith(".p")]
        poison_files = [os.path.join(poison_dir,f) for f in os.listdir(poison_dir) if f.endswith(".p")]

        np.random.shuffle(clean_files)
        np.random.shuffle(poison_files)

        # split clean and poison independently
        clean_train, clean_test = train_test_split(clean_files, test_size=0.2, random_state=67)
        poison_train, poison_test = train_test_split(poison_files, test_size=0.2, random_state=67)

        # get datasets
        train_dataset = BinaryPoisonDataset(clean_train, poison_train, poison_ratio=0.5)
        test_dataset = BinaryPoisonDataset(clean_test, poison_test, poison_ratio=self.poison_ratio)
        
        # data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ====================================== model config =========================================================
    def init_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2) # binary
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    # ====================================== training =========================================================
    def train_model(self, debug=False):
        epochs = self.num_epochs
        losses = []

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            if debug:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(self.train_loader):.4f}")
            losses.append(running_loss/len(self.train_loader))

        if debug:
            print("Training complete")
        
        return losses

    # ====================================== model eval =========================================================
    def eval_model(self, debug=False):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        # TN FP
        # FN TP
        if debug:
            print("\nConfusion Matrix:")
            print(cm)

        # more metrics
        # precision: TP / (TP + FP)
        # recall: TP / (TP + FN)
        report_dict = classification_report(all_labels, all_preds, target_names=["Clean", "Poison"], output_dict=True)
        
        if debug:
            report_str = classification_report(all_labels, all_preds, target_names=["Clean", "Poison"])
            print("\nClassification Report:")
            print(report_str)

        return report_dict, cm