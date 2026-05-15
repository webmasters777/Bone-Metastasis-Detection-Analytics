#!/usr/bin/env python3
"""Evaluate saved EfficientNet-B3 checkpoint and write results JSON."""
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Import utility functions and classes from training script
import train_efficientnet_b3 as trainer

BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "dataset project")
MODEL_PATH = os.path.join(BASE_DIR, "bone_scan_efficientnet_b3_final.pth")
RESULTS_PATH = os.path.join(BASE_DIR, "efficientnet_b3_results.json")
TRAINING_PATH = os.path.join(BASE_DIR, "efficientnet_b3_training.json")

if not os.path.exists(MODEL_PATH):
    print("Model checkpoint not found:", MODEL_PATH)
    sys.exit(1)

print("Loading dataset samples...")
image_paths, labels, per_view_counts = trainer.collect_dataset_samples(["RANT", "RPOST"], dataset_path=DATASET_PATH)
labels_array = np.array(labels)

# Recreate splits used during training
X_train_paths, X_test_paths, y_train, y_test = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# Transforms
test_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = trainer.BoneScanDataset(X_train_paths, y_train, transform=test_transform)
test_dataset = trainer.BoneScanDataset(X_test_paths, y_test, transform=test_transform)
full_dataset = trainer.BoneScanDataset(image_paths, labels, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
full_loader = DataLoader(full_dataset, batch_size=16, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Build model architecture and load weights
model = trainer.create_efficientnet_b3_model(weights_path=None)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)

# Evaluate
print("Running evaluation on test set...")
test_preds, test_labels = trainer.evaluate_loader(model, test_loader, device)
test_performance = trainer.compute_performance(test_preds, test_labels)

print("Running evaluation on full dataset...")
full_preds, full_labels = trainer.evaluate_loader(model, full_loader, device)
full_performance = trainer.compute_performance(full_preds, full_labels)

# Classification report and confusion matrix
try:
    test_report = classification_report(test_labels, test_preds, target_names=["Normal", "Metastasis"], output_dict=True, zero_division=0)
except Exception:
    test_report = {}

try:
    test_cm = confusion_matrix(test_labels, test_preds).tolist()
except Exception:
    test_cm = []

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
    "training": {},
    "performance_test": test_performance,
    "performance_full": full_performance,
    "confusion_matrix_test": test_cm,
    "classification_report_test": test_report,
    "weights": {
        "path": MODEL_PATH,
    },
}

# If training history exists, include summary
if os.path.exists(TRAINING_PATH):
    try:
        with open(TRAINING_PATH, 'r', encoding='utf-8') as fh:
            training = json.load(fh)
        results_data["training"] = {
            "epochs": training.get("epochs"),
            "final_train_loss": training.get("train_loss", [])[-1] if training.get("train_loss") else None,
            "final_val_loss": training.get("val_loss", [])[-1] if training.get("val_loss") else None,
            "best_val_accuracy": max(training.get("val_accuracy", [])) if training.get("val_accuracy") else None,
        }
    except Exception:
        pass

# Write results
with open(RESULTS_PATH, 'w', encoding='utf-8') as fh:
    json.dump(results_data, fh, indent=2)

print("Wrote results to", RESULTS_PATH)

# Print brief summary
print("Test performance:")
for k, v in test_performance.items():
    print(f"  {k}: {v}")

print("Full dataset performance:")
for k, v in full_performance.items():
    print(f"  {k}: {v}")
