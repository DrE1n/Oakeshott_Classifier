import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")


DATASET_ROOT = Path(r"D:\Oakeshott Project")
OUTPUT_DIR   = Path(r"D:\oakeshott_models")

# Mode:
# "coarse" — train main type classifier (pools all subtypes under parent)
# "fine"   — train subtype classifier for one specific type
MODE = "coarse"
#MODE = "fine"

# Only used when MODE == "fine"
# Set to the parent type to train a subtype classifier for
# e.g. "Type_XVIII" to train XVIIIa vs XVIIIb vs XVIIIc etc.
TYPE_FILTER = "Type_X"

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 80
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 3e-4
IMG_SIZE = 224

FREEZE_EPOCHS = 8

PATIENCE = 12

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResizeWithPadding:
    def __init__(self, size=IMG_SIZE):
        self.size = size

    def __call__(self, img):
        img.thumbnail((self.size, self.size), Image.LANCZOS)
        return ImageOps.pad(img, (self.size, self.size), color=(128, 128, 128))


train_transforms = transforms.Compose([
    ResizeWithPadding(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_test_transforms = transforms.Compose([
    ResizeWithPadding(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class SwordDataset(Dataset):

    def __init__(self, split: str, mode: str, type_filter: str = None,
                 transform=None):
        self.transform = transform
        self.samples = []  # list of (path, label_idx)
        self.classes = []  # ordered list of class names
        self.mode = mode

        split_dir = DATASET_ROOT / split

        if mode == "coarse":
            # Each parent type folder is one class
            # All images from all subtype subfolders are pooled
            type_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
            self.classes = [d.name for d in type_dirs]

            for label_idx, type_dir in enumerate(type_dirs):
                for subtype_dir in sorted(type_dir.iterdir()):
                    if not subtype_dir.is_dir():
                        continue
                    for img_path in subtype_dir.iterdir():
                        if img_path.suffix.lower() in VALID_EXTENSIONS:
                            self.samples.append((img_path, label_idx))

        elif mode == "fine":
            assert type_filter is not None, "type_filter required for fine mode"
            type_dir = split_dir / type_filter

            if not type_dir.exists():
                raise ValueError(f"Type folder not found: {type_dir}")

            subtype_dirs = sorted([d for d in type_dir.iterdir() if d.is_dir()])
            self.classes = [d.name for d in subtype_dirs]

            for label_idx, subtype_dir in enumerate(subtype_dirs):
                for img_path in subtype_dir.iterdir():
                    if img_path.suffix.lower() in VALID_EXTENSIONS:
                        self.samples.append((img_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def compute_class_weights(dataset):
    counts = torch.zeros(len(dataset.classes))
    for _, label in dataset.samples:
        counts[label] += 1
    weights = 1.0 / counts.clamp(min=1)
    # Dampen with sqrt to avoid overcorrecting
    weights = torch.sqrt(weights)
    weights = weights / weights.sum() * len(dataset.classes)
    return weights.to(device)


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model.to(device)


def freeze_backbone(model: nn.Module):
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True



def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = correct = total = 0

    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels



def plot_confusion_matrix(y_true, y_pred, classes, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(max(10, len(classes)), max(8, len(classes))))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {save_path}")



def train():
    # Output directory
    run_name = f"{MODE}" if MODE == "coarse" else f"fine_{TYPE_FILTER}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    print(f"\nLoading datasets (mode={MODE}" +
          (f", type={TYPE_FILTER}" if MODE == "fine" else "") + ")...")

    train_ds = SwordDataset("train", MODE, TYPE_FILTER, train_transforms)
    val_ds = SwordDataset("val", MODE, TYPE_FILTER, val_test_transforms)
    test_ds = SwordDataset("test", MODE, TYPE_FILTER, val_test_transforms)

    classes = train_ds.classes
    num_classes = len(classes)

    print(f"  Classes ({num_classes}): {classes}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Save class index mapping
    class_map = {i: name for i, name in enumerate(classes)}
    with open(run_dir / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)

    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = build_model(num_classes)
    print(f"\nModel: ResNet50 → {num_classes} classes")

    # Weighted loss for class imbalance
    class_weights = compute_class_weights(train_ds)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer — start with only FC trainable
    freeze_backbone(model)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Training loop
    best_val_loss = float("inf")
    patience_count = 0
    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}

    print(f"\nTraining for up to {NUM_EPOCHS} epochs "
          f"(backbone frozen for first {FREEZE_EPOCHS})...\n")

    for epoch in range(1, NUM_EPOCHS + 1):

        # Unfreeze backbone after FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS + 1:
            print("  Unfreezing backbone for full fine-tuning...")
            unfreeze_backbone(model)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE * 0.1,  # lower LR for pretrained layers
                weight_decay=WEIGHT_DECAY
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - FREEZE_EPOCHS
            )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
              f"Train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.3f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "classes": classes,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, run_dir / "best_model.pth")
            print(f"  ✓ New best saved (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    # Evaluation

    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(run_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion
    )

    print(f"\n{'=' * 50}")
    print(f"TEST RESULTS")
    print(f"{'=' * 50}")
    print(f"  Loss     : {test_loss:.4f}")
    print(f"  Accuracy : {test_acc:.4f} ({test_acc * 100:.1f}%)")
    print(f"\nPer-class report:")
    print(classification_report(test_labels, test_preds,
                                target_names=classes, digits=3))

    # Confusion matrix
    plot_confusion_matrix(test_labels, test_preds, classes,
                          run_dir / "confusion_matrix.png")

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"  Training curves saved to {run_dir / 'training_curves.png'}")

    # Save full results
    results = {
        "mode": MODE,
        "type_filter": TYPE_FILTER if MODE == "fine" else None,
        "classes": classes,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "best_epoch": checkpoint["epoch"],
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll outputs saved to {run_dir}")


if __name__ == "__main__":
    train()