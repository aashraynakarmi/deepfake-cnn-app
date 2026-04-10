import os
import cv2
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

def list_video_files(folder):
    if not os.path.isdir(folder):
        return []
    results = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        ext = os.path.splitext(name)[1].lower()
        if os.path.isfile(path) and ext in VIDEO_EXTENSIONS:
            results.append(path)
    return sorted(results)

def sample_frames_from_video(video_path, num_frames=8, resize=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        positions = list(range(num_frames))
    else:
        positions = np.linspace(0, max(0, total_frames - 1), num=num_frames, dtype=int).tolist()

    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames

class VideoFrameDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, frames_per_video=8):
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.samples = []

        for path in list_video_files(real_dir):
            self.samples.append((path, 0))
        for path in list_video_files(fake_dir):
            self.samples.append((path, 1))

        if not self.samples:
            raise ValueError("No videos found in dataset/real or dataset/fake.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_path, label = self.samples[index]
        frames = sample_frames_from_video(video_path, num_frames=self.frames_per_video)

        if not frames:
            tensor = torch.zeros((self.frames_per_video, 3, 128, 128), dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return tensor, label_tensor, video_path

        processed = []
        for frame in frames:
            if self.transform is not None:
                processed.append(self.transform(frame))
            else:
                arr = np.asarray(frame).astype("float32") / 255.0
                arr = np.transpose(arr, (2, 0, 1))
                processed.append(torch.tensor(arr, dtype=torch.float32))

        while len(processed) < self.frames_per_video:
            processed.append(processed[-1].clone())

        frames_tensor = torch.stack(processed[:self.frames_per_video], dim=0)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return frames_tensor, label_tensor, video_path
