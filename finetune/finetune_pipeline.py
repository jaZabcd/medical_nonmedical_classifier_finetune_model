import os
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import torch
from collections import Counter
import torch.nn as nn
import torch.optim as optim
import timm
import csv


# Config
TRAIN_DIR = "Dataset/train"
TEST_DIR = "Dataset/test"
BATCH_SIZE = 8
NUM_EPOCH = 10
LEARNING_RATE = 1e-4
PATIENCE = 5

OUTPUT_DIR = "training_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, 'efficientnetb0_finetuned.pth')

# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAutocontrast(0.5),
    transforms.RandomAdjustSharpness(0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),  # <-- REQUIRED before Normalize
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])




# === Load datasets ===
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)

# Count training class distribution
targets = train_dataset.targets
class_counts = Counter(targets)
print("Class distribution:", class_counts)


# Calculate weights for each class
total_samples = sum(class_counts.values())
class_weights = [total_samples / class_counts[i] for i in range(len(class_counts))]

print(f"Total sample : {total_samples}")
print(f"Class weights : {class_weights}")

# Assign weight to each sample
sample_weights = [class_weights[label] for label in targets]

print(f"sample weights : {len(sample_weights)}")

# 5. Use WeightedRandomSampler
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# 6. Replace train_loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)

# Keep test_loader unchanged
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Print to verify ===
print("Classes:", train_dataset.classes)
print("Number of training images:", len(train_dataset))
print("Number of test images:", len(test_dataset))

# MODEL
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# OPTIMIZER + SCHEDULER
weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


# TRAIN LOOP
def train():
    best_acc = 0.0
    early_stop_counter = 0

    with open(os.path.join(OUTPUT_DIR,"training_log.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc"])

    for epoch in range(NUM_EPOCH):
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        val_acc = evaluate()
        print(f"Epoch {epoch+1}/{NUM_EPOCH} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} - Val Acc: {val_acc:.4f}")

        with open("training_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, epoch_loss, epoch_acc.item(), val_acc])

        if val_acc > best_acc:
            best_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

# EVAL
@torch.no_grad()
def evaluate():
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total

if __name__ == '__main__':
    train()

