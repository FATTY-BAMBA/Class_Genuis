# app/app.py

import os
import uuid
import time
import redis
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from app.celery_app import celery as celery_app  # <-- single, shared Celery instance
# If you actually use runpod controls in handlers, keep these imports; otherwise you can remove them.
from app.runpod_controller import start_pod, get_pod_status  # noqa: F401

# ==================== Setup Base Directory & Environment ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '..', '.env'))

# ==================== Persistent Storage (e.g., RunPod network volume) ====================
PERSIST_BASE = os.getenv("PERSIST_BASE", "/workspace")
try:
    os.makedirs(PERSIST_BASE, exist_ok=True)
    _t = os.path.join(PERSIST_BASE, ".write_test")
    with open(_t, "w") as _f:
        _f.write("ok")
    os.remove(_t)
except Exception:
    # Fallback for local/dev containers without the volume mount
    PERSIST_BASE = "/app"

UPLOAD_FOLDER = os.path.join(PERSIST_BASE, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Optional: keep templates/static where they already are inside the image
TEMPLATES_DIR = "/app/templates"
STATIC_DIR = "/app/static"

# ==================== Flask App ====================
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB

# ==================== Redis (for health checks) ====================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.StrictRedis.from_url(REDIS_URL)

# ==================== Health Endpoint ====================
@app.route("/healthz")
def healthz():
    status = {"status": "ok", "redis": False, "celery": False}

    # Check Redis
    try:
        redis_client.ping()
        status["redis"] = True
    except Exception:
        status["status"] = "degraded"

    # Check Celery (uses the shared instance from app.celery_app)
    try:
        inspector = celery_app.control.inspect()
        ping = inspector.ping() if inspector else None
        status["celery"] = bool(ping)  # True if any worker responded
        if not status["celery"]:
            status["status"] = "degraded"
    except Exception:
        status["status"] = "degraded"

    return jsonify(status), 200 if status["status"] == "ok" else 503

# ==================== Routes ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Re-exported in tasks/tasks.py, so this import path is valid
    from tasks.tasks import process_video_task

    file = request.files.get('file')
    video_url = request.form.get('video_url')
    num_pages = int(request.form.get('num_pages', 3))
    num_questions = int(request.form.get('num_questions', 10))

    try:
        if file and file.filename:
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            video_url = f"file://{video_path}"
        elif not video_url:
            return jsonify({"error": "No file or video URL provided!"}), 400

        dummy_video_info = {
            "Id": str(uuid.uuid4()),
            "TeamId": "test",
            "SectionNo": "1",
            "CreatedAt": None
        }
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Enqueue with retries (handles momentary Redis hiccups)
    queued = False
    last_err = None
    for attempt in range(3):
        try:
            process_video_task.delay(video_url, dummy_video_info, num_questions, num_pages)
            queued = True
            break
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))

    if not queued:
        print(f"❌ Celery enqueue error after retries: {last_err}")
        return jsonify({"error": "❌ Redis queue unavailable, please try again later."}), 503

    return jsonify({
        "queued": True,
        "message": "✅ Upload received. Processing has started. The results will be sent to the 教學平台."
    }), 200

@app.route('/upload_url', methods=['POST'])
def upload_url():
    # Re-exported in tasks/tasks.py, so this import path is valid
    from tasks.tasks import process_video_task

    try:
        data = request.get_json(force=True)
        required_keys = {"Id", "TeamId", "SectionNo", "PlayUrl"}
        if not data or not required_keys.issubset(data.keys()):
            return jsonify({"error": f"Missing required keys: {required_keys}"}), 400

        # Optional knobs (fallback to task defaults if omitted)
        num_questions = int(data.get("NumQuestions", 10))
        num_pages = int(data.get("NumPages", 3))

        video_info = {
            "Id": data["Id"],
            "TeamId": data["TeamId"],
            "SectionNo": data["SectionNo"],
            "CreatedAt": data.get("CreatedAt")
        }
        play_url = data["PlayUrl"]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    queued = False
    last_err = None
    for attempt in range(3):
        try:
            process_video_task.delay(play_url, video_info, num_questions, num_pages)
            queued = True
            break
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))

    if not queued:
        print(f"❌ Celery enqueue error after retries: {last_err}")
        return jsonify({"error": "❌ Redis queue unavailable, please try again later."}), 503

    return jsonify({"status": "queued", "video": play_url}), 200
