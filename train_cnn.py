import os
import json
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

from dataset_utils import VideoFrameDataset
from model import SimpleDeepfakeCNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(BASE_DIR, "dataset", "real")
FAKE_DIR = os.path.join(BASE_DIR, "dataset", "fake")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "deepfake_cnn.pth")
META_PATH = os.path.join(MODELS_DIR, "metadata.json")

os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels, _paths in loader:
            b, t, c, h, w = frames.shape
            frames = frames.view(b * t, c, h, w).to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(frames).view(b, t)
            video_logits = logits.mean(dim=1)
            loss = criterion(video_logits, labels)

            probs = torch.sigmoid(video_logits)
            preds = (probs >= 0.5).float()

            total_loss += loss.item() * b
            correct += (preds == labels).sum().item()
            total += b

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc

def main():
    dataset = VideoFrameDataset(
        real_dir=REAL_DIR,
        fake_dir=FAKE_DIR,
        transform=TRANSFORM,
        frames_per_video=8,
    )

    total_videos = len(dataset)
    if total_videos < 4:
        raise ValueError("Add more videos. At least 4 total videos are recommended.")

    train_size = max(1, int(0.8 * total_videos))
    val_size = total_videos - train_size
    if val_size == 0:
        train_size = total_videos - 1
        val_size = 1

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    model = SimpleDeepfakeCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 8
    best_val_acc = 0.0
    history = []

    print(f"Training on {DEVICE}")
    print(f"Total videos: {total_videos} | Train: {train_size} | Val: {val_size}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for frames, labels, _paths in train_loader:
            b, t, c, h, w = frames.shape
            frames = frames.view(b * t, c, h, w).to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(frames).view(b, t)
            video_logits = logits.mean(dim=1)
            loss = criterion(video_logits, labels)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(video_logits)
            preds = (probs >= 0.5).float()

            running_loss += loss.item() * b
            total += b
            correct += (preds == labels).sum().item()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        })

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    metadata = {
        "model_type": "SimpleDeepfakeCNN",
        "image_size": [128, 128],
        "frames_per_video": 8,
        "best_val_accuracy": round(best_val_acc, 4),
        "device_used": str(DEVICE),
        "total_videos": total_videos,
        "train_videos": train_size,
        "val_videos": val_size,
        "history": history,
        "label_map": {"0": "Real", "1": "Deepfake"},
        "notes": "Video prediction is based on averaging frame-level CNN logits."
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved metadata to: {META_PATH}")

if __name__ == "__main__":
    main()
