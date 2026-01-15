=======
# Software_To_Hardware__Migration
Image Processing model is implemented into hardware 

![img1](project_snapshot.png)

---
# 🥭 Mango Plant Disease Detection (Raspberry Pi & PC Compatible)

## 📌 Project Overview
This project is an **offline Mango Plant Disease Detection system** designed to run reliably on **Raspberry Pi (32‑bit OS)** as well as **Windows/Linux PCs**.
It uses a **lightweight machine‑learning pipeline (feature extraction + SVM)** instead of heavy deep‑learning runtimes, ensuring **stability, compatibility, and field usability**.

The system supports:
- Image upload
- Live camera capture
- Disease prediction with confidence
- Cause, treatment, and prevention guidance
- Optional **voice output** with safe fallback (no crashes)

---

## 🌱 Diseases Supported
- Anthracnose
- Bacterial Canker
- Powdery Mildew
- Die Back
- Sooty Mould
- Gall Midge
- Cutting Weevil
- Healthy

---

## 🧠 System Architecture
```
Web Browser
   ↓
Flask Web Server
   ↓
Image Input (Upload / Camera)
   ↓
Feature Extraction (CPU-based)
   ↓
SVM Classifier
   ↓
JSON Output + Optional Voice
```

---
![img1](pi_installation.png)
---

## 🧩 Hardware Used
- Raspberry Pi 3B / 4B (ARMv7, 32‑bit)
- USB Camera / Pi Camera
- Bluetooth / Wired Speaker (optional)
- Development PC (Intel x64)

---

## 💻 Software & Versions
| Component | Version |
|---------|--------|
| Raspberry Pi OS | 32‑bit |
| Python | 3.9 – 3.11 |
| Flask | 2.x |
| OpenCV | 4.x |
| NumPy | 1.23+ |
| scikit‑learn | 1.2+ |
| pyttsx3 | Latest |

---

## 🚧 Problems Faced & Solutions
- ONNX incompatibility → replaced with SVM pipeline
- Feature mismatch → aligned training & inference vectors
- Audio crashes → non‑blocking silent TTS
- Image cache issues → cache‑buster preview

---

## 🚀 Run Command
```bash
python opt/mango-pi/server.py
```

---

### 👨‍💻 Author
**AI-Solution - KK**

