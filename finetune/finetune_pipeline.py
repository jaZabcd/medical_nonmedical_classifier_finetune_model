import os
from torchvision import transforms, datasets


# Config
DATASET_PATH = "../Dataset"
BATCH_SIZE = 8
NUM_EPOCH = 10
LEARNING_RATE = 1e-4
PATIENCE = 5

OUTPUT_PATH = "..//training_output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_PATH, 'efficientnetb0_finetuned.pth')

train_transform = transforms.Compose(
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAutocontrast(0.5),
    transforms.RandomAdjustSharpness(0.5),
    transforms.ColorJitter(brightness=0.1,contrast=0.1),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
)


val_transform = transforms.Compose(
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAutocontrast(0.5),
    transforms.RandomAdjustSharpness(0.5),
    transforms.ColorJitter(brightness=0.1,contrast=0.1),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
)

# Dataset
full_dataset = datasets.ImageFolder(DATASET_PATH)





