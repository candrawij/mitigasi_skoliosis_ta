"""
T3.1 EXP-01: EfficientNet-B0 Image Baseline.

Trains EfficientNet-B0 (pretrained ImageNet) on:
- Project Design curated images (5 classes)
- Sitting Posture Detection images (4 classes)

Uses PyTorch + torchvision.
"""
import json
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, balanced_accuracy_score, confusion_matrix,
                             classification_report)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "07_results" / "experiments"

# Dataset configs
DATASETS = {
    "project_design": {
        "image_dir": PROJECT_ROOT / "02_data" / "interim" / "project_design_curated",
        "split_csv": PROJECT_ROOT / "03_metadata" / "final_split" / "project_design.csv",
        "classes": ["leaning_backward", "leaning_forward", "leaning_left", "leaning_right", "upright"],
        "exp_id": "EXP-PD-CNN",
    },
    "sitting_posture_detection": {
        "image_dir": PROJECT_ROOT / "02_data" / "raw" / "Sitting Posture Detection.v2i.coco",
        "split_csv": PROJECT_ROOT / "03_metadata" / "final_split" / "sitting_posture_detection.csv",
        "classes": ["good_posture", "leaning_backward", "leaning_forward", "slouch"],
        "exp_id": "EXP-SPD-CNN",
    },
}

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
IMG_SIZE = 224
NUM_WORKERS = 0  # Windows compatibility


class PostureDataset(Dataset):
    """Custom dataset for posture classification images."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Return a blank image if loading fails
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(is_train=False):
    """Get image transforms."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def load_project_design_data(config, split_name):
    """Load Project Design images from curated directory."""
    split_df = pd.read_csv(config["split_csv"])
    split_data = split_df[split_df["split"] == split_name]

    image_paths = []
    labels = []
    class_to_idx = {c: i for i, c in enumerate(config["classes"])}

    for _, row in split_data.iterrows():
        class_name = row["class_name"]
        if class_name not in class_to_idx:
            continue
        # Images are stored in label subdirectories
        img_dir = config["image_dir"] / class_name
        filename = row.get("filename", "")
        if filename:
            img_path = img_dir / filename
        else:
            # Try to find by image_id
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = img_dir / (row["image_id"] + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            else:
                continue

        if img_path.exists():
            image_paths.append(str(img_path))
            labels.append(class_to_idx[class_name])

    return image_paths, labels


def load_spd_data(config, split_name):
    """Load Sitting Posture Detection images from COCO directory."""
    split_df = pd.read_csv(config["split_csv"])
    split_data = split_df[split_df["split"] == split_name]

    # Load COCO annotations to get class labels per image
    image_classes = {}
    for orig_split in ["train", "valid", "test"]:
        ann_file = config["image_dir"] / orig_split / "_annotations.coco.json"
        if not ann_file.exists():
            continue
        with open(ann_file) as f:
            coco = json.load(f)
        images_map = {img["id"]: img["file_name"] for img in coco["images"]}
        cats = {cat["id"]: cat["name"] for cat in coco["categories"]}
        for ann in coco["annotations"]:
            fname = images_map.get(ann["image_id"], "")
            cat = cats.get(ann["category_id"], "")
            if fname:
                image_classes[fname] = cat

    image_paths = []
    labels = []
    class_to_idx = {c: i for i, c in enumerate(config["classes"])}

    for _, row in split_data.iterrows():
        image_id = row["image_id"]
        class_name = image_classes.get(image_id, "")
        if class_name not in class_to_idx:
            continue

        # Find the actual file
        for orig_split in ["train", "valid", "test"]:
            img_path = config["image_dir"] / orig_split / image_id
            if img_path.exists():
                image_paths.append(str(img_path))
                labels.append(class_to_idx[class_name])
                break

    return image_paths, labels


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(all_targets) if all_targets else 0
    return avg_loss, np.array(all_preds), np.array(all_targets)


def run_experiment(dataset_key):
    config = DATASETS[dataset_key]
    exp_dir = RESULTS_DIR / config["exp_id"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  EXP-01: EfficientNet-B0 — {dataset_key}")
    print(f"{'#'*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load data
    if dataset_key == "project_design":
        load_fn = load_project_design_data
    else:
        load_fn = load_spd_data

    train_paths, train_labels = load_fn(config, "train")
    valid_paths, valid_labels = load_fn(config, "valid")
    test_paths, test_labels = load_fn(config, "test")

    print(f"  Train: {len(train_paths)}, Valid: {len(valid_paths)}, Test: {len(test_paths)}")

    if not train_paths or not test_paths:
        print("  ERROR: No data loaded!")
        return None

    # Create datasets
    train_ds = PostureDataset(train_paths, train_labels, get_transforms(is_train=True))
    valid_ds = PostureDataset(valid_paths, valid_labels, get_transforms(is_train=False))
    test_ds = PostureDataset(test_paths, test_labels, get_transforms(is_train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Create model
    n_classes = len(config["classes"])
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    best_valid_acc = 0
    best_epoch = 0
    history = []

    print(f"\n  Training for {NUM_EPOCHS} epochs...")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_preds, valid_targets = evaluate_model(model, valid_loader, criterion, device)
        valid_acc = accuracy_score(valid_targets, valid_preds)

        scheduler.step(valid_loss)

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "valid_loss": round(valid_loss, 4),
            "valid_acc": round(valid_acc, 4),
            "lr": optimizer.param_groups[0]["lr"],
        })

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), exp_dir / "best_model.pt")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}")

    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed:.1f}s")
    print(f"  Best valid acc: {best_valid_acc:.4f} at epoch {best_epoch}")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(exp_dir / "best_model.pt", weights_only=True))
    test_loss, test_preds, test_targets = evaluate_model(model, test_loader, criterion, device)

    class_names = config["classes"]
    acc = accuracy_score(test_targets, test_preds)
    bal_acc = balanced_accuracy_score(test_targets, test_preds)
    avg = "macro"
    prec = precision_score(test_targets, test_preds, average=avg, zero_division=0)
    rec = recall_score(test_targets, test_preds, average=avg, zero_division=0)
    f1 = f1_score(test_targets, test_preds, average=avg, zero_division=0)
    cm = confusion_matrix(test_targets, test_preds)

    print(f"\n{'='*60}")
    print(f"  TEST RESULTS — {dataset_key}")
    print(f"{'='*60}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Precision (macro): {prec:.4f}")
    print(f"  Recall (macro):    {rec:.4f}")
    print(f"  F1 (macro):        {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")
    print(f"\n{classification_report(test_targets, test_preds, target_names=class_names, zero_division=0)}")

    results = {
        "dataset": dataset_key,
        "model": "EfficientNet-B0",
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "best_epoch": best_epoch,
        "best_valid_acc": round(best_valid_acc, 4),
        "training_time_s": round(elapsed, 1),
        "num_epochs": NUM_EPOCHS,
        "timestamp": datetime.now().isoformat(),
    }

    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(history).to_csv(exp_dir / "training_history.csv", index=False)

    return results


def main():
    all_results = []

    # Run on Project Design
    pd_results = run_experiment("project_design")
    if pd_results:
        all_results.append(pd_results)

    # Run on SPD
    spd_results = run_experiment("sitting_posture_detection")
    if spd_results:
        all_results.append(spd_results)

    # Summary
    if all_results:
        print(f"\n{'#'*60}")
        print("  EXP-01 SUMMARY")
        print(f"{'#'*60}")
        for r in all_results:
            print(f"  {r['dataset']:30s} | Acc={r['accuracy']:.4f} | F1={r['f1_macro']:.4f} "
                  f"| Best epoch={r['best_epoch']}")

        with open(RESULTS_DIR / "exp01_all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
