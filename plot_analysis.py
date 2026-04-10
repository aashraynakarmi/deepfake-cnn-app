import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------- 1. Correlation Matrix ----------
features_csv = os.path.join(BASE_DIR, "dataset_features.csv")
if os.path.exists(features_csv):
    df = pd.read_csv(features_csv)

    numeric_cols = ["frames", "fps", "width", "height", "duration_sec"]
    available_cols = [c for c in numeric_cols if c in df.columns]

    if available_cols:
        numeric_df = df[available_cols]
        corr = numeric_df.corr()

        plt.figure(figsize=(7, 5))
        plt.imshow(corr, interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Correlation Matrix of Video Features")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "correlation_matrix.png"))
        plt.close()

    if "label" in df.columns:
        plt.figure(figsize=(6, 4))
        class_counts = df["label"].value_counts()
        plt.bar(class_counts.index.astype(str), class_counts.values)
        plt.title("Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Number of Videos")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"))
        plt.close()

# ---------- 2. Accuracy Graph from metadata ----------
meta_path = os.path.join(BASE_DIR, "models", "metadata.json")
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    history = metadata.get("history", [])
    if history:
        epochs = [x["epoch"] for x in history]
        train_acc = [x["train_accuracy"] for x in history]
        val_acc = [x["val_accuracy"] for x in history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
        plt.plot(epochs, val_acc, marker="o", label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy Graph")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "model_accuracy_graph.png"))
        plt.close()

# ---------- 3. Evaluation Plots ----------
eval_path = os.path.join(BASE_DIR, "evaluation_report.json")
if os.path.exists(eval_path):
    with open(eval_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Confusion Matrix
    cm = report.get("confusion_matrix", None)
    if cm is not None:
        cm = np.array(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
        disp.plot()
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
        plt.close()

    # Metrics Bar Chart
    metrics = {
        "Accuracy": report.get("accuracy", 0),
        "Precision": report.get("precision", 0),
        "Recall": report.get("recall", 0),
        "F1-Score": report.get("f1_score", 0),
    }

    plt.figure(figsize=(7, 5))
    plt.bar(metrics.keys(), metrics.values())
    plt.ylim(0, 1)
    plt.title("Performance Metrics")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "performance_metrics.png"))
    plt.close()

    # ROC and PR Curves
    y_true = report.get("y_true", [])
    y_scores = report.get("y_scores", [])

    if y_true and y_scores and len(y_true) == len(y_scores):
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"CNN (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random Baseline")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"))
        plt.close()

        # Precision-Recall
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall_vals, precision_vals)

        plt.figure(figsize=(7, 5))
        plt.plot(recall_vals, precision_vals, label=f"CNN (AUC-PR = {pr_auc:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "precision_recall_curve.png"))
        plt.close()

print("Saved analysis graphs to plots/")