#!/usr/bin/env python3
"""
EfficientNet-B3 Training Script for Bone Metastasis Detection using PyTorch.
Trains a fine-tuned EfficientNet-B3 model on the bone scan dataset.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import EfficientNet_B3_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# Dataset path
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "dataset project")


def load_dataset_labels(view_type="RANT", dataset_path=DATASET_PATH):
    """Load labels from the dataset text file."""
    labels_file = os.path.join(dataset_path, f"chest{view_type}", f"chest{view_type}.txt")
    labels_dict = {}

    with open(labels_file, "r") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                filename, label = parts
                labels_dict[filename] = int(label)

    return labels_dict


def normalize_view_types(view_types):
    normalized = []
    for view in view_types:
        upper = view.strip().upper()
        if upper not in ("RANT", "RPOST"):
            raise ValueError(f"Unsupported view type: {view}")
        normalized.append(upper)
    return normalized


def collect_dataset_samples(view_types, dataset_path=DATASET_PATH):
    image_paths = []
    labels = []
    per_view_counts = {}

    for view in normalize_view_types(view_types):
        image_dir = os.path.join(dataset_path, f"chest{view}")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Dataset folder not found: {image_dir}")

        labels_dict = load_dataset_labels(view_type=view, dataset_path=dataset_path)
        view_count = 0

        for filename in sorted(os.listdir(image_dir)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if filename not in labels_dict:
                continue

            image_paths.append(os.path.join(image_dir, filename))
            labels.append(labels_dict[filename])
            view_count += 1

        per_view_counts[view] = view_count

    return image_paths, labels, per_view_counts


class BoneScanDataset(Dataset):
    """Custom dataset for bone scan images."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def resolve_local_weights(weights_path=None):
    if weights_path:
        return weights_path

    cache_name = "efficientnet_b3_rwightman-b3899882.pth"
    cache_path = os.path.expanduser(os.path.join("~", ".cache", "torch", "hub", "checkpoints", cache_name))
    if os.path.exists(cache_path):
        return cache_path

    project_path = os.path.join(BASE_DIR, cache_name)
    if os.path.exists(project_path):
        return project_path

    weights_dir = os.path.join(BASE_DIR, "weights")
    weights_path = os.path.join(weights_dir, cache_name)
    if os.path.exists(weights_path):
        return weights_path

    return None


def create_efficientnet_b3_model(weights_path=None):
    """Create and modify EfficientNet-B3 model."""
    local_weights = resolve_local_weights(weights_path)

    if local_weights:
        if not os.path.exists(local_weights):
            raise FileNotFoundError(f"EfficientNet-B3 weights not found: {local_weights}")
        model = models.efficientnet_b3(weights=None)
        state_dict = torch.load(local_weights, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
    else:
        model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last few feature blocks
    for param in model.features[-2:].parameters():
        param.requires_grad = True

    # Modify classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )

    return model


def compute_performance(predictions, ground_truth):
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    tn = np.sum((predictions == 0) & (ground_truth == 0))
    fp = np.sum((predictions == 1) & (ground_truth == 0))
    fn = np.sum((predictions == 0) & (ground_truth == 1))
    tp = np.sum((predictions == 1) & (ground_truth == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "f1_score": float(f1),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def evaluate_loader(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels_batch in data_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(images).squeeze().view(-1)
            preds = (outputs > 0.5).float()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels_batch.cpu().numpy().tolist())

    return all_preds, all_labels


def train_efficientnet_b3(view_types, num_epochs, dataset_path=DATASET_PATH, output_dir=BASE_DIR, weights_path=None):
    """Train EfficientNet-B3 model."""
    print("Loading dataset...")

    image_paths, labels, per_view_counts = collect_dataset_samples(view_types, dataset_path)
    labels_array = np.array(labels)

    print(f"Dataset loaded: {len(image_paths)} samples")
    print("Per-view counts:")
    for view, count in per_view_counts.items():
        print(f"  {view}: {count}")
    print(f"Normal: {np.sum(labels_array == 0)}, Metastasis: {np.sum(labels_array == 1)}")

    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Train: {len(X_train_paths)}, Test: {len(X_test_paths)}")

    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = BoneScanDataset(X_train_paths, y_train, transform=train_transform)
    test_dataset = BoneScanDataset(X_test_paths, y_test, transform=test_transform)
    full_dataset = BoneScanDataset(image_paths, labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    full_loader = DataLoader(full_dataset, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    sys.stdout.flush()
    
    model = create_efficientnet_b3_model(weights_path=weights_path).to(device)
    print("Model created and loaded")
    sys.stdout.flush()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    num_epochs = int(num_epochs)
    best_accuracy = 0.0
    model_path = os.path.join(output_dir, "bone_scan_efficientnet_b3_final.pth")

    history = {
        "epochs": list(range(1, num_epochs + 1)),
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "settings": {
            "epochs": num_epochs,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "loss": "BCELoss",
            "input_size": 300,
        },
    }

    print("Training EfficientNet-B3...")
    sys.stdout.flush()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, labels_batch in train_loader:
            images = images.to(device)
            labels_batch = labels_batch.float().to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze().view(-1)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels_batch in test_loader:
                images = images.to(device)
                labels_batch = labels_batch.to(device)

                outputs = model(images).squeeze().view(-1)
                loss = criterion(outputs, labels_batch.float())
                val_loss += loss.item()

                predicted = (outputs > 0.5).float()
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()

        accuracy = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(accuracy / 100)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%"
        )
        sys.stdout.flush()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_path)

    print(f"Best validation accuracy: {best_accuracy:.2f}%")

    # Save training history
    training_path = os.path.join(output_dir, "efficientnet_b3_training.json")
    with open(training_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    # Final evaluation on test set
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_preds, test_labels = evaluate_loader(model, test_loader, device)

    test_performance = compute_performance(test_preds, test_labels)
    test_report = classification_report(
        test_labels,
        test_preds,
        target_names=["Normal", "Metastasis"],
        output_dict=True,
        zero_division=0,
    )
    test_cm = confusion_matrix(test_labels, test_preds)

    # Full dataset evaluation
    full_preds, full_labels = evaluate_loader(model, full_loader, device)
    full_performance = compute_performance(full_preds, full_labels)

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "model": "EfficientNet-B3",
        "dataset": {
            "total": len(image_paths),
            "normal": int(np.sum(labels_array == 0)),
            "metastasis": int(np.sum(labels_array == 1)),
            "per_view": per_view_counts,
        },
        "split": {
            "train_size": len(X_train_paths),
            "test_size": len(X_test_paths),
            "test_fraction": 0.2,
        },
        "training": {
            "epochs": num_epochs,
            "best_val_accuracy": best_accuracy / 100,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "loss": "BCELoss",
            "input_size": 300,
        },
        "performance_test": test_performance,
        "performance_full": full_performance,
        "confusion_matrix_test": test_cm.tolist(),
        "classification_report_test": test_report,
        "weights": {
            "source": "local" if resolve_local_weights(weights_path) else "torchvision_pretrained",
            "path": resolve_local_weights(weights_path),
        },
    }

    results_path = os.path.join(output_dir, "efficientnet_b3_results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results_data, handle, indent=2)

    print("\nClassification Report (Test Set):")
    print(classification_report(test_labels, test_preds, target_names=["Normal", "Metastasis"], zero_division=0))

    print("\nConfusion Matrix (Test Set):")
    print(test_cm)

    print(f"\nModel saved to: {model_path}")
    print(f"Training history saved to: {training_path}")
    print(f"Results saved to: {results_path}")

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B3 on bone scan datasets.")
    parser.add_argument(
        "--views",
        nargs="+",
        default=["RANT", "RPOST"],
        help="Dataset views to include: RANT RPOST (default: both)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--dataset-path",
        default=DATASET_PATH,
        help="Path to dataset root (default: dataset project)",
    )
    parser.add_argument(
        "--output-dir",
        default=BASE_DIR,
        help="Where to save model and results (default: project root)",
    )
    parser.add_argument(
        "--weights-path",
        default="",
        help="Path to EfficientNet-B3 weights file (.pth). If omitted, uses torchvision weights.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    weights_path = args.weights_path.strip() or None
    train_efficientnet_b3(args.views, args.epochs, args.dataset_path, args.output_dir, weights_path)
