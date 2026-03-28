import os
from xml.parsers.expat import model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import medmnist
from medmnist import DermaMNIST
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import pandas as pd
from torch.utils.data import Subset
import random

# Automatically use GPU if available, otherwise fallback to CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_sample_images():
    """
    Plots one original DermaMNIST image per class (no augmentation/resize transforms).
    """
    info = medmnist.INFO['dermamnist']
    label_dict = info['label']
    class_indices = sorted(int(k) for k in label_dict.keys())
    samples_by_class = {}

    raw_dataset = DermaMNIST(split="val", transform=None, download=True, size=64)

    for i in range(len(raw_dataset)):
        img, label = raw_dataset[i]

        if torch.is_tensor(label):
            class_idx = int(label.squeeze().item())
        else:
            class_idx = int(np.asarray(label).squeeze())

        if class_idx not in samples_by_class:
            samples_by_class[class_idx] = img

        if len(samples_by_class) == len(class_indices):
            break

    if not samples_by_class:
        raise ValueError("No samples found in the raw DermaMNIST dataset.")

    cols = min(4, len(class_indices))
    rows = int(np.ceil(len(class_indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = np.array(axes).reshape(-1)

    for i, class_idx in enumerate(class_indices):
        class_name = label_dict.get(str(class_idx), f"Class {class_idx}")

        if class_idx in samples_by_class:
            img = samples_by_class[class_idx]
            img_np = np.asarray(img)
            axes[i].imshow(img_np)
        else:
            axes[i].text(0.5, 0.5, "No sample\nfound", ha='center', va='center', fontsize=10)

        formatted_title = class_name.replace(" ", "\n") if len(class_name) > 15 else class_name
        axes[i].set_title(formatted_title, fontsize=11, fontweight='bold')
        axes[i].axis('off')

    for j in range(len(class_indices), len(axes)):
        axes[j].axis('off')

    missing_classes = [idx for idx in class_indices if idx not in samples_by_class]
    if missing_classes:
        print(f"Warning: no sample found for class indices {missing_classes}.")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------
def load_dermamnist(batch_size=128, download=True):
    """Loads DermaMNIST, applies transforms, and returns dataloaders and label dictionary."""
    info = medmnist.INFO['dermamnist']
    label_dict = info['label']
    
    # Standard ImageNet normalization for ResNet
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    data_transform_train = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    data_transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    train_dataset = DermaMNIST(split='train', transform=data_transform_train, download=download, size=28)
    val_dataset = DermaMNIST(split='val', transform=data_transform_val_test, download=download, size=28)
    test_dataset = DermaMNIST(split='test', transform=data_transform_val_test, download=download, size=28)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader, label_dict, train_dataset, val_dataset, test_dataset


def get_small_train_loader(train_dataset, num_samples=50, batch_size=10, seed=42):
    """
    Creates a very small training loader to demonstrate overfitting.
    Takes a random subset of the training data.
    """
    # Create a random subset of indices
    random.seed(seed)
    np.random.seed(seed)
    indices = list(range(len(train_dataset)))

    # take only indices where the label is 5
    # indices = [i for i in indices if train_dataset[i][1].item() == 5]

    random.shuffle(indices)
    subset_indices = indices[:num_samples]
    
    # Create the small dataset and loader
    small_dataset = Subset(train_dataset, subset_indices)
    small_loader = torch.utils.data.DataLoader(dataset=small_dataset, batch_size=batch_size, shuffle=True)
    
    return small_loader, small_dataset

# ---------------------------------------------------------
# 2. MODEL BUILDING
# ---------------------------------------------------------
def get_model(num_classes=7):
    """Returns a pre-trained ResNet18 with a frozen backbone and a multi-layer classification head."""
    # 1. Load the pre-trained model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # 2. Freeze the backbone
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. Build the new Non-Linear Classification Head
    num_ftrs = model.fc.in_features  # For ResNet-18, this is 512
    hidden_dim = num_ftrs * 2        # Double the dimension to 1024
    
    # ugly hack to replace the final layer with a non-linear head
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, num_classes)
    )
    return model.to(DEVICE)

# ---------------------------------------------------------
# 3. TRAINING LOOP (The "Black Box")
# ---------------------------------------------------------
def train_model(model, train_loader, val_loader, train_dataset, epochs=3, learning_rate=1e-4, use_class_weights=False):
    """Trains the model. Abstracts away gradients, backprop, and device management."""
    
    # Class Imbalance Handling (Median Frequency Balancing)
    if use_class_weights:
        class_labels = train_dataset.labels.squeeze()
        count_labels = np.bincount(class_labels)
        median_freq = np.median(count_labels)
        weights = median_freq / count_labels
        class_weights = torch.FloatTensor(weights).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Class weights applied.")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Training WITHOUT class weights.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.squeeze().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        
        # Simple Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.squeeze().to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    print("✅ Training complete.")
    return model

# ---------------------------------------------------------
# 4. EVALUATION & VISUALIZATION
# ---------------------------------------------------------
def evaluate_and_plot_confusion_matrix(model, test_loader, label_dict):
    """Evaluates the model on the test set and plots a confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())

    # Get string labels
    display_labels = [label_dict[str(i)] for i in range(len(label_dict))]

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Clinical Confusion Matrix")
    plt.tight_layout()
    plt.show()


def evaluate_specific_class(model, data_loader, target_class_idx, label_dict):
    """Evaluates the model's accuracy specifically on a single target class."""
    model.eval()
    class_correct = 0
    class_total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.squeeze().to(DEVICE)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Isolate predictions where the true label is our target class
            mask = (targets == target_class_idx)
            class_total += mask.sum().item()
            class_correct += (predicted[mask] == targets[mask]).sum().item()

    if class_total > 0:
        class_acc = class_correct / class_total
        print(f"Model Accuracy strictly on Class {target_class_idx} ({label_dict[str(target_class_idx)]}): {class_acc * 100:.2f}%")
    else:
        print(f"No samples found for Class {target_class_idx} in this data split.")



def get_number_of_samples_per_class_as_table(train_dataset, label_dict):
    """Prints a table of the number of samples per class in the training dataset."""
    class_labels = train_dataset.labels.squeeze()
    count_labels = np.bincount(class_labels)
    
    data = {
        "Class Index": list(label_dict.keys()),
        "Class Name": [label_dict[key] for key in label_dict.keys()],
        "Number of Samples": count_labels
    }
    
    df = pd.DataFrame(data)
    # drop 
    return df