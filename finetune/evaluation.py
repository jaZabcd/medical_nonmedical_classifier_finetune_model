from torchvision import transforms, datasets
import timm
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Paths
MODEL_PATH = "training_output/efficientnetb0_finetuned.pth"
TEST_DIR = "Dataset/test"

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Transforms
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset and DataLoader
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=inference_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Class names
class_names = test_dataset.classes  # ['medical', 'non_medical']

def plot_confusion_matrix(cm, class_names, filename="training_output/confusion_matrix.png"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_classification_report(report_dict, filename="training_output/classification_report.png"):
    metrics = ['precision', 'recall', 'f1-score']
    class_labels = [k for k in report_dict.keys() if k in ['medical', 'non_medical']]

    values = {
        metric: [report_dict[label][metric] for label in class_labels]
        for metric in metrics
    }

    x = np.arange(len(class_labels))
    width = 0.25

    plt.figure(figsize=(8, 5))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, values[metric], width, label=metric)

    plt.xticks(x + width, class_labels)
    plt.ylim(0, 1)
    plt.title("Classification Report Metrics")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def evaluate_detailed(model, dataloader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, class_names)

    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    plot_classification_report(report)

    # Optionally save raw report to a text file
    with open("training_output/classification_report.txt", "w") as f:
        f.write(classification_report(all_labels, all_preds, target_names=class_names))

# Run evaluation
evaluate_detailed(model, test_loader, device, class_names)
