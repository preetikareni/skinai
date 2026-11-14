import os
import json
import base64
import io
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from chat import chat_bp

# Try to import tensorflow - if not available give a helpful error
try:
    from tflite_runtime.interpreter import Interpreter
    MODEL_LOADED = True
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        MODEL_LOADED = True
    except ImportError:
        MODEL_LOADED = False
        print("⚠️  No TFLite interpreter found.")

# Try to import PIL
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  Pillow not found. Install with: pip install Pillow")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.register_blueprint(chat_bp)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "skin_model.tflite")
CLASS_MAP_PATH  = os.path.join(BASE_DIR, "model", "class_indices.json")
RECO_PATH       = os.path.join(BASE_DIR, "recommendations.json")

# Must match IMG_SIZE in train.py
IMG_SIZE = (224, 224)

# ─── Load Assets ──────────────────────────────────────────────────────────────
model = None
class_map = {}
recommendations = {}

def load_assets():
    global model, class_map, recommendations

    # Load recommendations
    if os.path.exists(RECO_PATH):
        with open(RECO_PATH, "r", encoding="utf-8") as f:
            recommendations = json.load(f)
        print(f"✅ Recommendations loaded: {list(recommendations.keys())}")
    else:
        print(f"❌ recommendations.json not found at {RECO_PATH}")

    # Load class map
    if os.path.exists(CLASS_MAP_PATH):
        with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
            class_map = json.load(f)
        print(f"✅ Class map loaded: {class_map}")
    else:
        # Fallback: use dataset folder names sorted alphabetically
        dataset_path = os.path.join(BASE_DIR, "dataset")
        if os.path.isdir(dataset_path):
            folders = sorted([
                d for d in os.listdir(dataset_path)
                if os.path.isdir(os.path.join(dataset_path, d))
            ])
            class_map = {str(i): name for i, name in enumerate(folders)}
            print(f"⚠️  No class_indices.json found. Using fallback: {class_map}")
        else:
            print("❌ Could not determine class map.")

    # Load model
    if MODEL_LOADED and os.path.exists(MODEL_PATH):
        try:
            model = Interpreter(model_path=MODEL_PATH)
            model.allocate_tensors()
            print(f"✅ Model loaded from: {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            model = None
    else:
        if not MODEL_LOADED:
            print("❌ TensorFlow not installed.")
        else:
            print(f"⚠️  Model file not found at {MODEL_PATH}")
            print("   Run: python train.py  to train the model first.")


def predict_skin(image_bytes):
    """Predict skin condition from image bytes."""
    if model is None:
        return None, None, "Model not loaded. Please run train.py first."

    try:
        # Load image and resize to the same size used in training (224x224)
        if PIL_AVAILABLE:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = img.resize(IMG_SIZE)
            img_array = np.array(img, dtype=np.float32)
        else:
            import cv2
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            img_array = img.astype(np.float32)

        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

        # Same preprocessing as train.py (scales pixels to -1..1)
        img_array = img_array / 127.5 - 1.0
        inp = model.get_input_details()[0]
        out = model.get_output_details()[0]
        model.set_tensor(inp["index"], img_array.astype(np.float32))
        model.invoke()
        predictions = model.get_tensor(out["index"])[0]
        class_idx = int(np.argmax(predictions))
        confidence = float(predictions[class_idx])
        label = class_map.get(str(class_idx), f"class_{class_idx}")

        # All class confidences
        all_probs = {
            class_map.get(str(i), f"class_{i}"): float(p)
            for i, p in enumerate(predictions)
        }

        return label, confidence, all_probs

    except Exception as e:
        return None, None, f"Prediction error: {str(e)}"


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    model_ready = model is not None
    return render_template("index.html", model_ready=model_ready)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename."}), 400

    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"success": False, "error": f"Unsupported format: {ext}"}), 400

    image_bytes = file.read()

    # Predict
    label, confidence, all_probs = predict_skin(image_bytes)

    if label is None:
        return jsonify({"success": False, "error": str(all_probs)}), 500

    # Get recommendations
    reco = recommendations.get(label, {})

    # Encode image as base64 for display
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    return jsonify({
        "success": True,
        "label": label,
        "confidence": round(confidence * 100, 1),
        "all_probs": {k: round(v * 100, 1) for k, v in all_probs.items()},
        "recommendations": reco,
        "image_data": f"data:{mime};base64,{b64_image}",
    })


@app.route("/status")
def status():
    return jsonify({
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "classes": list(class_map.values()),
        "tensorflow": MODEL_LOADED,
    })


load_assets()   # runs on gunicorn AND locally

if __name__ == "__main__":
    print("\n🌟 Skin AI Analysis Server Starting...")
    print("   Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)