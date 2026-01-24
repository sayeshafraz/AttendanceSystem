import os
import json
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageOps
import requests

from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename


# =========================
# FLASK CONFIG
# =========================
app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(STATIC_DIR, "results")

MODELS_DIR = os.path.join(BASE_DIR, "models")

STUDENT_DB_DIR = os.path.join(BASE_DIR, "student_db")
STUDENT_IMG_DIR = os.path.join(STUDENT_DB_DIR, "images")
STUDENT_DB_JSON = os.path.join(STUDENT_DB_DIR, "students.json")

CURRENT_GROUP_PATH = os.path.join(UPLOAD_DIR, "group.png")
ANNOTATED_PATH = os.path.join(RESULTS_DIR, "annotated.png")
LAST_RESULTS_JSON = os.path.join(UPLOAD_DIR, "last_results.json")

# cache models
_YUNET_CACHE = None
_SFACE_CACHE = None


# =========================
# MODELS CONFIG (YuNet + SFace)
# =========================
YUNET_URL = (
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/"
    "face_detection_yunet_2023mar.onnx?download=true"
)
SFACE_URL = (
    "https://huggingface.co/opencv/face_recognition_sface/resolve/main/"
    "face_recognition_sface_2021dec.onnx?download=true"
)

YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

MATCH_THRESHOLD = 0.36
YUNET_SCORE_THRESHOLD = 0.45
YUNET_NMS_THRESHOLD = 0.30
YUNET_TOPK = 5000


# =========================
# BASIC HELPERS
# =========================
def ensure_folder(path: str):
    os.makedirs(path, exist_ok=True)


# =========================
# API LOGIN (TOKEN ONLY)
# =========================
BASE_URL = "https://api.slcloud.3em.tech"
ACCESS_TOKEN = None     # server-side saved token
LOGGED_EMAIL = None     # optional (for showing in UI)


def login_get_token(email: str, password: str):
    """
    Returns (token_or_none, message, status_code)
    """
    url = f"{BASE_URL}/api/Auth/login"
    payload = {"email": email, "password": password}

    try:
        r = requests.post(url, json=payload, headers={"accept": "text/plain"}, timeout=30)
    except Exception as e:
        return None, f"API connection failed: {e}", 500

    # wrong creds etc
    if r.status_code >= 400:
        # show a nicer message (don’t expose sensitive info)
        return None, "Invalid email or password.", 401

    # success
    try:
        data = r.json()
    except Exception:
        return None, "Login response is not valid JSON.", 500

    token = data.get("accessToken")
    if not token:
        return None, "Token not found in response.", 500

    return token, "Login successful.", 200


def file_exists(p: str) -> bool:
    return bool(p) and os.path.isfile(p)


def is_valid_image_path(p: str) -> bool:
    if not file_exists(p):
        return False
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def is_uploaded_file_an_image(file_storage) -> bool:
    """
    Accept ALL extensions, but validate the uploaded file is actually an image.
    """
    try:
        file_storage.stream.seek(0)
        im = Image.open(file_storage.stream)
        im.verify()
        file_storage.stream.seek(0)
        return True
    except Exception:
        try:
            file_storage.stream.seek(0)
        except Exception:
            pass
        return False


def safe_name(text: str) -> str:
    if text is None:
        return "unknown"
    s = str(text).strip().replace(" ", "_")
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
    return "".join(out) if out else "unknown"


def load_image_bgr_any_format(path: str):
    """Load image via PIL (fix EXIF rotation) and return OpenCV BGR."""
    with Image.open(path) as pil_img:
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert("RGB")
        rgb = np.array(pil_img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def save_png_normalized(src_path: str, out_path: str):
    """Save image as PNG with EXIF fixed."""
    ensure_folder(os.path.dirname(out_path))
    with Image.open(src_path) as pil_img:
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert("RGB")
        pil_img.save(out_path, format="PNG")


# =========================
# STUDENT DB
# =========================
def load_student_db():
    ensure_folder(STUDENT_DB_DIR)
    ensure_folder(STUDENT_IMG_DIR)

    if not os.path.isfile(STUDENT_DB_JSON):
        return []

    try:
        with open(STUDENT_DB_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            cleaned = []
            for st in data:
                if isinstance(st, dict) and "roll" in st and "name" in st and "img_path" in st:
                    cleaned.append(st)
            return cleaned
        return []
    except Exception:
        return []


def save_student_db(students_list):
    ensure_folder(STUDENT_DB_DIR)
    with open(STUDENT_DB_JSON, "w", encoding="utf-8") as f:
        json.dump(students_list, f, indent=2, ensure_ascii=False)


def save_student_image_permanent(src_path: str, roll: str, name: str):
    ensure_folder(STUDENT_IMG_DIR)
    r = safe_name(roll)
    n = safe_name(name)
    filename = f"{r}_{n}.png"
    out_path = os.path.join(STUDENT_IMG_DIR, filename)
    save_png_normalized(src_path, out_path)
    return out_path


def delete_student_by_roll(roll: str):
    roll = str(roll).strip()
    if not roll:
        return False, "Roll is empty."

    db = load_student_db()
    if not db:
        return False, "No saved students."

    kept = []
    deleted_img = None
    found = False

    for st in db:
        if str(st.get("roll", "")).strip() == roll:
            found = True
            deleted_img = st.get("img_path", "")
        else:
            kept.append(st)

    if not found:
        return False, "Roll not found."

    save_student_db(kept)

    if deleted_img and os.path.isfile(deleted_img):
        try:
            os.remove(deleted_img)
        except Exception as e:
            return False, f"Removed from JSON, but could not delete image: {e}"

    return True, "Student removed."


# =========================
# MODEL DOWNLOADER (ROBUST)
# =========================
def download_file(url: str, out_path: str, timeout: int = 90):
    ensure_folder(os.path.dirname(out_path))
    tmp_path = out_path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp_path, "wb") as f:
        f.write(r.read())
    os.replace(tmp_path, out_path)


def ensure_models():
    ensure_folder(MODELS_DIR)
    if not os.path.isfile(YUNET_PATH):
        download_file(YUNET_URL, YUNET_PATH)
    if not os.path.isfile(SFACE_PATH):
        download_file(SFACE_URL, SFACE_PATH)


def load_yunet_sface():
    global _YUNET_CACHE, _SFACE_CACHE

    if not (hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")):
        raise RuntimeError("Install: pip install --upgrade opencv-contrib-python")

    # IMPORTANT: lazy download (not at startup)
    ensure_models()

    if _YUNET_CACHE is None:
        _YUNET_CACHE = cv2.FaceDetectorYN.create(
            YUNET_PATH,
            "",
            (320, 320),
            score_threshold=YUNET_SCORE_THRESHOLD,
            nms_threshold=YUNET_NMS_THRESHOLD,
            top_k=YUNET_TOPK,
        )

    if _SFACE_CACHE is None:
        _SFACE_CACHE = cv2.FaceRecognizerSF.create(SFACE_PATH, "")

    return _YUNET_CACHE, _SFACE_CACHE


# =========================
# FACE PIPELINE
# =========================
def detect_faces(img_bgr):
    detector, _ = load_yunet_sface()
    H, W = img_bgr.shape[:2]

    max_side = max(H, W)
    target_max = 1280
    scale = 1.0

    if max_side > target_max:
        scale = target_max / float(max_side)
        newW = int(W * scale)
        newH = int(H * scale)
        img_small = cv2.resize(img_bgr, (newW, newH), interpolation=cv2.INTER_AREA)
    else:
        img_small = img_bgr
        newH, newW = H, W

    detector.setInputSize((newW, newH))
    _, faces = detector.detect(img_small)

    if faces is not None and len(faces) > 0 and scale != 1.0:
        faces_scaled = faces.copy()
        faces_scaled[:, 0] = faces[:, 0] / scale
        faces_scaled[:, 1] = faces[:, 1] / scale
        faces_scaled[:, 2] = faces[:, 2] / scale
        faces_scaled[:, 3] = faces[:, 3] / scale
        for i in range(5):
            faces_scaled[:, 5 + 2 * i] = faces[:, 5 + 2 * i] / scale
            faces_scaled[:, 6 + 2 * i] = faces[:, 6 + 2 * i] / scale
        faces = faces_scaled

    return faces


def pick_largest_face(faces):
    if faces is None or len(faces) == 0:
        return None
    areas = [float(f[2] * f[3]) for f in faces]
    return faces[int(np.argmax(areas))]


def align_crop_face(img_bgr, face_row):
    _, recognizer = load_yunet_sface()
    return recognizer.alignCrop(img_bgr, face_row)


def feature_from_aligned(aligned_face_bgr):
    _, recognizer = load_yunet_sface()
    return recognizer.feature(aligned_face_bgr)


def cosine_sim(feat_a, feat_b):
    _, recognizer = load_yunet_sface()
    return float(recognizer.match(feat_a, feat_b, cv2.FaceRecognizerSF_FR_COSINE))


def greedy_assign(sim_matrix, threshold):
    S, G = sim_matrix.shape
    pairs = [(float(sim_matrix[i, j]), i, j) for i in range(S) for j in range(G)]
    pairs.sort(reverse=True, key=lambda x: x[0])

    used_s = set()
    used_g = set()
    assign = {}

    for sim, si, gi in pairs:
        if sim < threshold:
            break
        if si in used_s or gi in used_g:
            continue
        assign[si] = gi
        used_s.add(si)
        used_g.add(gi)

    return assign


def draw_present_outlines(group_img_bgr, group_faces, matched_group_indexes):
    annotated = group_img_bgr.copy()
    if group_faces is None:
        return annotated

    for gi, f in enumerate(group_faces):
        if gi not in matched_group_indexes:
            continue
        x, y, w, h = map(int, f[:4])
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return annotated


def run_recognition_from_saved_students(group_path):
    db_students = load_student_db()
    students = []
    for st in db_students:
        students.append(
            {
                "name": st.get("name", ""),
                "roll": st.get("roll", ""),
                "img_path": st.get("img_path", ""),
                "status": "Absent",
            }
        )

    if not students:
        return students, None, "No saved students. Add students first."

    if not file_exists(group_path) or not is_valid_image_path(group_path):
        return students, None, "Group image invalid."

    group_img = load_image_bgr_any_format(group_path)
    group_faces = detect_faces(group_img)

    if group_faces is None or len(group_faces) == 0:
        return students, group_img, "No faces in group."

    group_feats = []
    for f in group_faces:
        g_aligned = align_crop_face(group_img, f)
        gf = feature_from_aligned(g_aligned)
        group_feats.append(gf)

    valid_idx = []
    student_feats = []

    for i, st in enumerate(students):
        spath = st.get("img_path", "")
        if not file_exists(spath) or not is_valid_image_path(spath):
            st["status"] = "Invalid"
            continue

        stu_img = load_image_bgr_any_format(spath)
        stu_faces = detect_faces(stu_img)
        stu_face = pick_largest_face(stu_faces)

        if stu_face is None:
            st["status"] = "Invalid"
            continue

        s_aligned = align_crop_face(stu_img, stu_face)
        sf = feature_from_aligned(s_aligned)

        valid_idx.append(i)
        student_feats.append(sf)

    if not valid_idx:
        return students, group_img, "No valid student faces in saved images."

    S = len(student_feats)
    G = len(group_feats)
    sim_matrix = np.zeros((S, G), dtype=np.float32)

    for si in range(S):
        for gi in range(G):
            sim_matrix[si, gi] = cosine_sim(student_feats[si], group_feats[gi])

    assignments = greedy_assign(sim_matrix, MATCH_THRESHOLD)

    for local_si in range(S):
        orig_i = valid_idx[local_si]
        students[orig_i]["status"] = "Present" if local_si in assignments else "Absent"

    matched_group_indexes = set(assignments.values())
    annotated = draw_present_outlines(group_img, group_faces, matched_group_indexes)

    return students, annotated, "Done"


# =========================
# ROUTES
# =========================
@app.get("/")
def index():
    return render_template("attendenceapp.html")


# NEW: login route (frontend will send email+password)
@app.post("/login")
def login():
    global ACCESS_TOKEN, LOGGED_EMAIL

    email = (request.form.get("email") or "").strip()
    password = (request.form.get("password") or "").strip()

    if not email or not password:
        return {"ok": False, "message": "Email and password are required."}, 400

    token, msg, code = login_get_token(email, password)

    if not token:
        return {"ok": False, "message": msg}, code

    ACCESS_TOKEN = token
    LOGGED_EMAIL = email
    return {"ok": True, "message": "Login successful."}, 200


@app.get("/api/state")
def api_state():
    ensure_folder(UPLOAD_DIR)
    ensure_folder(RESULTS_DIR)
    ensure_folder(STUDENT_DB_DIR)
    ensure_folder(STUDENT_IMG_DIR)

    db = load_student_db()

    # apply statuses from last run (if exists)
    status_map = {}
    if os.path.isfile(LAST_RESULTS_JSON):
        try:
            with open(LAST_RESULTS_JSON, "r", encoding="utf-8") as f:
                rows = json.load(f) or []
            for r in rows:
                status_map[str(r.get("roll", "")).strip()] = r.get("status", "Saved")
        except Exception:
            status_map = {}

    students = []
    for st in db:
        img_path = st.get("img_path", "")
        roll = str(st.get("roll", "")).strip()
        students.append(
            {
                "name": st.get("name", ""),
                "roll": roll,
                "img_file": os.path.basename(img_path) if img_path else "",
                "status": status_map.get(roll, "Saved"),
            }
        )

    group_exists = os.path.isfile(CURRENT_GROUP_PATH) and is_valid_image_path(CURRENT_GROUP_PATH)
    annotated_exists = os.path.isfile(ANNOTATED_PATH)

    return {
        "students": students,
        "group_exists": group_exists,
        "annotated_exists": annotated_exists,
        "message": "",
        # extra (won't break old UI)
        "logged_in": bool(ACCESS_TOKEN),
        "logged_email": LOGGED_EMAIL or ""
    }


@app.get("/group_image")
def group_image():
    if not os.path.isfile(CURRENT_GROUP_PATH):
        return ("No group image", 404)
    return send_from_directory(UPLOAD_DIR, "group.png")


@app.get("/student_image/<path:filename>")
def student_image(filename):
    return send_from_directory(STUDENT_IMG_DIR, filename)


@app.post("/upload_group")
def upload_group():
    ensure_folder(UPLOAD_DIR)

    f = request.files.get("group_image")
    if not f or f.filename == "":
        return {"message": "Please select a group image."}, 400

    # ACCEPT ALL EXTENSIONS BUT VALIDATE REAL IMAGE
    if not is_uploaded_file_an_image(f):
        return {"message": "This file is not a valid image."}, 400

    tmp_name = secure_filename(f.filename) or "group_upload"
    tmp_path = os.path.join(UPLOAD_DIR, tmp_name)
    f.save(tmp_path)

    try:
        save_png_normalized(tmp_path, CURRENT_GROUP_PATH)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return {"message": "Group image uploaded."}
    except Exception as e:
        return {"message": f"Failed to process group image: {e}"}, 500


@app.post("/add_student")
def add_student():
    ensure_folder(UPLOAD_DIR)
    ensure_folder(STUDENT_DB_DIR)
    ensure_folder(STUDENT_IMG_DIR)

    name = (request.form.get("name") or "").strip()
    roll = (request.form.get("roll") or "").strip()
    f = request.files.get("student_image")

    if not name or not roll:
        return {"message": "Name and Roll are required."}, 400

    if not f or f.filename == "":
        return {"message": "Please select a student image."}, 400

    # ACCEPT ALL EXTENSIONS BUT VALIDATE REAL IMAGE
    if not is_uploaded_file_an_image(f):
        return {"message": "This file is not a valid image."}, 400

    tmp_name = secure_filename(f.filename) or "student_upload"
    tmp_path = os.path.join(UPLOAD_DIR, tmp_name)
    f.save(tmp_path)

    try:
        saved_path = save_student_image_permanent(tmp_path, roll, name)
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        db = load_student_db()
        db = [x for x in db if str(x.get("roll", "")).strip() != roll]  # overwrite same roll
        db.append({"name": name, "roll": roll, "img_path": saved_path})
        save_student_db(db)

        return {"message": f"Student saved: {os.path.basename(saved_path)}"}
    except Exception as e:
        return {"message": f"Failed to add student: {e}"}, 500


@app.post("/delete_student/<roll>")
def delete_student(roll):
    ok, msg = delete_student_by_roll(roll)
    return {"message": msg if ok else f"FAILED: {msg}"}, (200 if ok else 400)


@app.post("/run")
def run():
    # do recognition + save annotated + save last_results.json
    try:
        ensure_folder(UPLOAD_DIR)
        ensure_folder(RESULTS_DIR)

        rows, annotated, msg = run_recognition_from_saved_students(CURRENT_GROUP_PATH)

        # save rows
        with open(LAST_RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        # save annotated
        if annotated is not None:
            cv2.imwrite(ANNOTATED_PATH, annotated)

        return {"message": msg}
    except Exception as e:
        return {"message": f"FAILED: {e}"}, 500


if __name__ == "__main__":
    ensure_folder(UPLOAD_DIR)
    ensure_folder(RESULTS_DIR)
    ensure_folder(STUDENT_IMG_DIR)

    # NO AUTO LOGIN HERE - user logs in from the form
    app.run(host="127.0.0.1", port=5000, debug=True)
