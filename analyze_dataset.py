import os
import cv2
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REAL_DIR = os.path.join(DATASET_DIR, "real")
FAKE_DIR = os.path.join(DATASET_DIR, "fake")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

def list_videos(folder):
    if not os.path.exists(folder):
        return []
    files = []
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        ext = os.path.splitext(f)[1].lower()
        if os.path.isfile(path) and ext in VIDEO_EXTENSIONS:
            files.append(path)
    return sorted(files)

def get_video_info(video_path, label):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = 0
    if fps and fps > 0:
        duration = frame_count / fps

    cap.release()

    return {
        "file": os.path.basename(video_path),
        "label": label,
        "frames": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": duration
    }

def analyze_class(folder, label):
    videos = list_videos(folder)
    infos = []
    corrupted = []

    for v in videos:
        info = get_video_info(v, label)
        if info is None:
            corrupted.append(os.path.basename(v))
        else:
            infos.append(info)

    if infos:
        avg_frames = np.mean([x["frames"] for x in infos])
        avg_fps = np.mean([x["fps"] for x in infos if x["fps"] > 0])
        avg_width = np.mean([x["width"] for x in infos])
        avg_height = np.mean([x["height"] for x in infos])
        avg_duration = np.mean([x["duration_sec"] for x in infos])
    else:
        avg_frames = avg_fps = avg_width = avg_height = avg_duration = 0

    return {
        "summary": {
            "label": label,
            "total_videos": len(videos),
            "readable_videos": len(infos),
            "corrupted_videos": len(corrupted),
            "avg_frames": round(float(avg_frames), 2),
            "avg_fps": round(float(avg_fps), 2),
            "avg_width": round(float(avg_width), 2),
            "avg_height": round(float(avg_height), 2),
            "avg_duration_sec": round(float(avg_duration), 2),
            "corrupted_files": corrupted
        },
        "rows": infos
    }

def main():
    real_stats = analyze_class(REAL_DIR, "real")
    fake_stats = analyze_class(FAKE_DIR, "fake")

    summary = {
        "real": real_stats["summary"],
        "fake": fake_stats["summary"],
        "total_videos": real_stats["summary"]["total_videos"] + fake_stats["summary"]["total_videos"]
    }

    print(json.dumps(summary, indent=2))

    with open("dataset_analysis.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rows = real_stats["rows"] + fake_stats["rows"]
    df = pd.DataFrame(rows)
    df.to_csv("dataset_features.csv", index=False)

    print("\nSaved dataset analysis to dataset_analysis.json")
    print("Saved per-video features to dataset_features.csv")

if __name__ == "__main__":
    main()