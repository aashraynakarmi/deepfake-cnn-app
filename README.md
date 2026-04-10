# Deepfake CNN Video Detection App

This project trains a CNN on **real** and **fake** videos, then serves a Flask web app for prediction.

## Dataset format

Place your videos like this:

```text
deepfake_cnn_app/
  dataset/
    real/
      original_001.mp4
      original_002.mp4
    fake/
      deepfake_001.mp4
      deepfake_002.mp4
```

For FaceForensics++, copy a subset of the original videos into `dataset/real` and a subset of manipulated videos into `dataset/fake`.

## Recommended demo advice

FaceForensics++ is large. For a professor demo on a laptop, use a **small subset** first, for example:
- 20 to 50 real videos
- 20 to 50 fake videos

That will train much faster and still demonstrate the CNN pipeline clearly.

## Run on Mac

Use **Python 3.10 or 3.11** if possible.

### 1. Open Terminal
### 2. Go to the project folder
```bash
cd ~/Downloads/deepfake_cnn_app
```

### 3. Create a virtual environment
```bash
python3 -m venv venv
```

### 4. Activate it
```bash
source venv/bin/activate
```

### 5. Upgrade pip
```bash
python3 -m pip install --upgrade pip
```

### 6. Install dependencies
```bash
pip install -r requirements.txt
```

### 7. Add dataset videos
Put videos into:
- `dataset/real`
- `dataset/fake`

### 8. Train the CNN
```bash
python3 train_cnn.py
```

### 9. Run the web app
```bash
python3 app.py
```

### 10. Open in browser
```text
http://127.0.0.1:5000
```

## Notes

- The model is a simple 2D CNN trained on sampled video frames.
- Video prediction is made by averaging the frame-level outputs.
- This is a strong academic demo, but not a production forensic system.
