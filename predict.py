import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from dataset_utils import sample_frames_from_video
from model import SimpleDeepfakeCNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "deepfake_cnn.pth")
META_PATH = os.path.join(BASE_DIR, "models", "metadata.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def model_available():
    return os.path.exists(MODEL_PATH)

def _load_model():
    model = SimpleDeepfakeCNN().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def _load_meta():
    if not os.path.exists(META_PATH):
        return {"frames_per_video": 8}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def predict_video(video_path):
    if not model_available():
        raise FileNotFoundError("Train the model first.")

    metadata = _load_meta()
    frames_per_video = int(metadata.get("frames_per_video", 8))
    frames = sample_frames_from_video(video_path, num_frames=frames_per_video)

    if not frames:
        return {
            "label": "Error",
            "probability_fake": 0.0,
            "confidence": 0.0,
            "summary": "The uploaded video could not be decoded.",
            "frame_count_used": 0,
        }

    tensors = []
    for frame in frames:
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(frame)
        tensors.append(TRANSFORM(frame))

    batch = torch.stack(tensors, dim=0).to(DEVICE)

    model = _load_model()
    with torch.no_grad():
        logits = model(batch).squeeze(1)
        probs = torch.sigmoid(logits)
        mean_prob = float(probs.mean().item())

    label = "Deepfake" if mean_prob >= 0.5 else "Real"
    confidence = mean_prob if mean_prob >= 0.5 else 1.0 - mean_prob

    if label == "Deepfake":
        summary = "The CNN found frame patterns more similar to fake/manipulated videos."
    else:
        summary = "The CNN found frame patterns more similar to real/original videos."

    return {
        "label": label,
        "probability_fake": round(mean_prob * 100, 2),
        "confidence": round(confidence * 100, 2),
        "summary": summary,
        "frame_count_used": len(tensors),
    }
