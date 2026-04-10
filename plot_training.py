import os
import json
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(BASE_DIR, "models", "metadata.json")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

history = metadata.get("history", [])

epochs = [x["epoch"] for x in history]
train_loss = [x["train_loss"] for x in history]
val_loss = [x["val_loss"] for x in history]
train_acc = [x["train_accuracy"] for x in history]
val_acc = [x["val_accuracy"] for x in history]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker="o", label="Train Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "loss_curve.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
plt.plot(epochs, val_acc, marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "accuracy_curve.png"))
plt.close()

print("Saved plots to plots/loss_curve.png and plots/accuracy_curve.png")