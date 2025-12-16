"""
Face Recognition using PyTorch Deep Learning
Implements a CNN for the ORL face dataset
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torchvision.transforms as transforms


# Custom Dataset Class
class FaceDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] - 1  # Convert labels to 0-indexed

        # Convert to tensor and add channel dimension
        image = torch.from_numpy(image).float().unsqueeze(0) / 255.0

        if self.transform:
            image = self.transform(image)

        return image, label

# Load images function
def load_images_from_folder(folder_path, target_size=(112, 92)):
    images = []
    labels = []
    for subdir in sorted(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subdir)
        if not os.path.isdir(subfolder_path):
            continue
        if subdir[1:].isdigit():
            label = int(subdir[1:])
        else:
            print(f"Skipping folder: {subdir}")
            continue

        for filename in sorted(os.listdir(subfolder_path)):
            file_path = os.path.join(subfolder_path, filename)
            if not file_path.endswith(".pgm"):
                continue
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            resized_image = cv2.resize(image, target_size)
            images.append(resized_image)
            labels.append(label)

    return np.array(images), np.array(labels)

# Load dataset
dataset_folder = "drive/MyDrive/ORL"
images, labels = load_images_from_folder(dataset_folder)
print(f"Loaded {len(images)} images with {len(np.unique(labels))} unique labels")

# Data augmentation transforms
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
])

# Create datasets
full_dataset = FaceDataset(images, labels, transform=None)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define CNN Model
class FaceRecognitionCNN(nn.Module):
    def __init__(self, num_classes=40):
        super(FaceRecognitionCNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 11, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = FaceRecognitionCNN(num_classes=40).to(device)
print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(loader), 100 * correct / total

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(loader), 100 * correct / total, all_preds, all_labels

# Training loop
num_epochs = 50
train_losses, train_accs = [], []
val_losses, val_accs = [], []
best_val_acc = 0.0

print("\nStarting training...")
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, _, _ = validate(model, test_loader, criterion, device)

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}] '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_face_model_pytorch.pth')
        print(f'✓ Saved best model with validation accuracy: {val_acc:.2f}%')

# Load best model
model.load_state_dict(torch.load('best_face_model_pytorch.pth'))

# Final evaluation
_, final_acc, y_pred, y_test = validate(model, test_loader, criterion, device)
print(f'\nFinal Test Accuracy: {final_acc:.2f}%')

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(train_accs, label='Train Accuracy')
ax2.plot(val_accs, label='Val Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('pytorch_training_history.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualize predictions
model.eval()
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
axes = axes.ravel()

# Get some test samples
test_samples = []
for images, labels in test_loader:
    test_samples = list(zip(images[:15], labels[:15]))
    break

with torch.no_grad():
    for idx, (image, true_label) in enumerate(test_samples):
        if idx >= 15:
            break

        image_input = image.unsqueeze(0).to(device)
        output = model(image_input)
        _, predicted = torch.max(output, 1)
        pred_label = predicted.item()

        axes[idx].imshow(image.squeeze().cpu().numpy(), cmap='gray')
        axes[idx].axis('off')
        color = 'green' if pred_label == true_label.item() else 'red'
        axes[idx].set_title(f'True: {true_label.item()+1}\nPred: {pred_label+1}',
                           color=color, fontsize=10)

plt.tight_layout()
plt.savefig('pytorch_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nModel saved as 'best_face_model_pytorch.pth'")