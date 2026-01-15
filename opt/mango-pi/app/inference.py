# =================================================
# FILE: app/inference.py
# Torch-free | TFLite | 512-dim embeddings
# =================================================

import os
import pickle
import numpy as np
import cv2
from PIL import Image

import tensorflow as tf

# =================================================
# PATHS
# =================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    ROOT,
    "models",
    "efficientnetv2_b0_embedding_512.tflite"
)

EMB_DIR = os.path.join(ROOT, "embeddings_cache")

SVC_PATH = os.path.join(EMB_DIR, "svc_model.pkl")
CENTROIDS_PATH = os.path.join(EMB_DIR, "centroids.npy")
CLASSES_PATH = os.path.join(EMB_DIR, "classes.npy")

# =================================================
# LOAD TFLITE MODEL
# =================================================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =================================================
# LOAD CLASSIFIER DATA
# =================================================
with open(SVC_PATH, "rb") as f:
    svm = pickle.load(f)

centroids = np.load(CENTROIDS_PATH)
classes = np.load(CLASSES_PATH)

# =================================================
# DISEASE INFO
# =================================================
DISEASE_TREATMENT = {
    "Anthracnose": {
        "cause": "Fungal infection causing dark lesions.",
        "treatment": "Spray Carbendazim 0.1%",
        "prevention": "Avoid excess humidity"
    },
    "Bacterial Canker": {
        "cause": "Bacterial infection causing cracks.",
        "treatment": "Copper fungicide + Streptocycline",
        "prevention": "Use disease-free plants"
    },
    "Healthy": {
        "cause": "No disease detected.",
        "treatment": "No treatment required",
        "prevention": "Maintain orchard hygiene"
    }
}

# =================================================
# LEAF PRESENCE CHECK (FAST & PI SAFE)
# =================================================
def is_leaf_present(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    img = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.count_nonzero(mask) / mask.size

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    return green_ratio > 0.15 and sharpness > 20

# =================================================
# PREPROCESS (MATCH TRAINING)
# =================================================
def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.asarray(img, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std
    img = np.expand_dims(img, axis=0)

    return img

# =================================================
# FEATURE EXTRACTION (512 DIM)
# =================================================
def extract_embedding(image_path):
    x = preprocess(image_path)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    emb = interpreter.get_tensor(output_details[0]["index"])[0]
    emb = emb.astype(np.float32)
    emb = emb / np.linalg.norm(emb)

    return emb  # shape: (512,)

# =================================================
# MAIN PREDICTION
# =================================================
def predict_image(image_path):

    if not is_leaf_present(image_path):
        return {
            "status": "rejected",
            "predicted_label": "No Leaf Detected",
            "confidence": 0.0,
            "cause": "No leaf detected",
            "treatment": "Capture a clear mango leaf",
            "prevention": "Use plain background"
        }

    emb = extract_embedding(image_path)

    # ---------- Open-set check (centroid distance)
    d = np.min(np.linalg.norm(centroids - emb, axis=1))
    if d > 1.2:
        return {
            "status": "rejected",
            "predicted_label": "Unknown Leaf",
            "confidence": 0.0,
            "cause": "Not a mango leaf",
            "treatment": "Only mango leaves supported",
            "prevention": "Use correct leaf"
        }

    # ---------- SVM classification
    if hasattr(svm, "predict_proba"):
        probs = svm.predict_proba([emb])[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
    else:
        idx = int(svm.predict([emb])[0])
        confidence = 0.99

    label = str(classes[idx])
    info = DISEASE_TREATMENT.get(label, DISEASE_TREATMENT["Healthy"])

    return {
        "status": "success",
        "predicted_label": label,
        "confidence": round(confidence, 4),
        "cause": info["cause"],
        "treatment": info["treatment"],
        "prevention": info["prevention"]
    }