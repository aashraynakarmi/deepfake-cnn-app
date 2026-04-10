import os
import json
import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset_utils import VideoFrameDataset
from model import SimpleDeepfakeCNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(BASE_DIR, "dataset", "real")
FAKE_DIR = os.path.join(BASE_DIR, "dataset", "fake")
MODEL_PATH = os.path.join(BASE_DIR, "models", "deepfake_cnn.pth")
REPORT_PATH = os.path.join(BASE_DIR, "evaluation_report.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keep this same as training
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def main():
    dataset = VideoFrameDataset(
        real_dir=REAL_DIR,
        fake_dir=FAKE_DIR,
        transform=TRANSFORM,
        frames_per_video=12,
    )

    total_videos = len(dataset)
    train_size = max(1, int(0.8 * total_videos))
    val_size = total_videos - train_size
    if val_size == 0:
        train_size = total_videos - 1
        val_size = 1

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    model = SimpleDeepfakeCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for frames, labels, _paths in val_loader:
            b, t, c, h, w = frames.shape
            frames = frames.view(b * t, c, h, w).to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(frames).view(b, t)
            video_logits = logits.mean(dim=1)
            probs = torch.sigmoid(video_logits)
            preds = (probs >= 0.5).float()

            y_true.extend(labels.cpu().numpy().astype(int).tolist())
            y_pred.extend(preds.cpu().numpy().astype(int).tolist())
            y_scores.extend(probs.cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=["Real", "Fake"], zero_division=0
    )

    roc_auc = roc_auc_score(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    print("Accuracy :", round(acc, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))
    print("ROC AUC  :", round(roc_auc, 4))
    print("AP Score :", round(avg_precision, 4))
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    results = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "average_precision": round(float(avg_precision), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": [round(float(x), 6) for x in y_scores],
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved evaluation to {REPORT_PATH}")

if __name__ == "__main__":
    main()