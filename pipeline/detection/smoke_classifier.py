"""
MobileNetV2 binary classifier: clear (0) vs hazy (1).
Trained on DeSmoke-LAP dataset (Pan et al. 2022).

Dataset structure expected:
    data/desmoke_lap/TLH_*/clear/*.jpg
    data/desmoke_lap/TLH_*/hazy/*.jpg

Train:  python -m pipeline.detection.smoke_classifier --data data/desmoke_lap --epochs 15
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image


# Dataset

class DeSmokeLAPDataset(Dataset):
    """
    Loads clear/ and hazy/ frames from all TLH_* video folders.
    Skips test_clip/ — that is held out for evaluation only.
    Labels: clear=0, hazy=1
    """

    def __init__(self, root: str, transform=None):
        self.samples = []
        self.transform = transform

        for video_dir in sorted(os.listdir(root)):
            video_path = os.path.join(root, video_dir)
            if not os.path.isdir(video_path):
                continue

            for label, subfolder in [(0, "clear"), (1, "hazy")]:
                subfolder_path = os.path.join(video_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue
                for fname in os.listdir(subfolder_path):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append(
                            (os.path.join(subfolder_path, fname), label)
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# Model

TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # mild only — color fidelity matters
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

TRANSFORM_EVAL = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_classifier(pretrained: bool = True) -> nn.Module:
    """
    MobileNetV2 with frozen backbone, trainable classifier head.
    Unfreeze backbone after initial head training if accuracy plateaus.
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    # Freeze backbone — only head trains in first pass
    for p in model.features.parameters():
        p.requires_grad = False

    return model


# Training

def train(data_root: str, save_path: str, epochs: int = 15,
          batch_size: int = 32, lr: float = 1e-3, val_split: float = 0.15):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = DeSmokeLAPDataset(data_root, transform=TRANSFORM_TRAIN)
    print(f"Total samples loaded: {len(dataset)}")

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Eval split should use TRANSFORM_EVAL — patch transform after split
    val_ds.dataset.transform = TRANSFORM_EVAL

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    model = build_classifier(pretrained=True).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss, train_correct, n = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * len(imgs)
            train_correct += (out.argmax(1) == labels).sum().item()
            n += len(imgs)

        # Validate
        model.eval()
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_n += len(imgs)

        val_acc = val_correct / val_n
        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"train_loss={train_loss/n:.4f}  train_acc={train_correct/n:.3f}  "
            f"val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model (val_acc={val_acc:.3f})")

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.3f}")
    print(f"Weights saved to: {save_path}")


def fine_tune(data_root: str, load_path: str, save_path: str,
              epochs: int = 10, batch_size: int = 32, lr: float = 1e-4):
    """
    Stage 2 training: unfreeze full backbone and fine-tune end-to-end.
    Must be run AFTER train() has saved a head-trained checkpoint.
    Uses lower LR to avoid destroying pretrained features.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Fine-tuning on device: {device}")

    dataset = DeSmokeLAPDataset(data_root, transform=TRANSFORM_TRAIN)
    val_size = int(len(dataset) * 0.15)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    val_ds.dataset.transform = TRANSFORM_EVAL

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=(device=="cuda"))

    # Load the head-trained checkpoint
    model = build_classifier(pretrained=False)
    model.load_state_dict(torch.load(load_path, map_location=device))

    # Unfreeze entire network
    for p in model.parameters():
        p.requires_grad = True

    # Differential LR: lower for backbone, higher for head
    optimizer = torch.optim.Adam([
        {"params": model.features.parameters(), "lr": lr},
        {"params": model.classifier.parameters(), "lr": lr * 10},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, n = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * len(imgs)
            train_correct += (out.argmax(1) == labels).sum().item()
            n += len(imgs)

        scheduler.step()

        model.eval()
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_correct += (model(imgs).argmax(1) == labels).sum().item()
                val_n += len(imgs)

        val_acc = val_correct / val_n
        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"train_loss={train_loss/n:.4f}  train_acc={train_correct/n:.3f}  "
            f"val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model (val_acc={val_acc:.3f})")

    print(f"\nFine-tuning complete. Best val_acc: {best_val_acc:.3f}")

# Inference

def load_classifier(weights_path: str, device: str = None) -> nn.Module:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_classifier(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model


def predict_frame(model: nn.Module, frame, device: str = None) -> tuple[int, float]:
    """
    frame: BGR uint8 numpy array (OpenCV format)
    Returns (label: int, confidence: float)
    label: 0=clear, 1=hazy
    """
    if device is None:
        device = next(model.parameters()).device
    img = Image.fromarray(frame[..., ::-1])  # BGR → RGB
    tensor = TRANSFORM_EVAL(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    label = int(probs.argmax(1).item())
    confidence = float(probs[0, label].item())
    return label, confidence


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ucl_laparoscopic_dataset")
    parser.add_argument("--save", default="weights/smoke_classifier_finetuned.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--finetune", action="store_true",
                        help="Run stage 2 fine-tuning from existing checkpoint")
    parser.add_argument("--load",     default="weights/smoke_classifier.pth",
                        help="Checkpoint to load for fine-tuning")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    if args.finetune:
        fine_tune(args.data, args.load, args.save,
                  epochs=args.epochs, batch_size=args.batch)
    else:
        train(args.data, args.save, epochs=args.epochs, batch_size=args.batch)