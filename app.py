import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from predict import predict_video, model_available

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = "/tmp/uploads"
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 700 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    video_url = None

    if request.method == "POST":
        if not model_available():
            error = "Model not found. Train the CNN first by running: python3 train_cnn.py"
            return render_template("index.html", result=result, error=error, video_url=video_url)

        if "video" not in request.files:
            error = "No file was uploaded."
        else:
            file = request.files["video"]
            if file.filename == "":
                error = "Please choose a video file."
            elif not allowed_file(file.filename):
                error = "Unsupported format. Use mp4, mov, avi, or mkv."
            else:
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(save_path)
                result = predict_video(save_path)
                video_url = None

    return render_template("index.html", result=result, error=error, video_url=video_url)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
