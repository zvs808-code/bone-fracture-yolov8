# 🦴 Bone Fracture Detector — YOLOv8 + Optuna AutoML

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![mAP50](https://img.shields.io/badge/mAP50-90.4%25-brightgreen)](README.md)
[![ONNX](https://img.shields.io/badge/ONNX-1.20-blue)](https://onnxruntime.ai/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow)](https://github.com/ultralytics/ultralytics)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Netlify-00C7B7)](https://bone-fracture-yolov8.netlify.app)

> On-device YOLOv8 bone fracture detection — runs in any browser, on Android, iOS, macOS, and Windows with **no server required**.

**🌐 Live Demo: [https://bone-fracture-yolov8.netlify.app](https://bone-fracture-yolov8.netlify.app)**  
**📦 Source: [https://github.com/zvs808-code/bone-fracture-yolov8](https://github.com/zvs808-code/bone-fracture-yolov8)**

---

## 📊 Results

| Model | Size | mAP50 | mAP50-95 | Deployment |
|---|---|---|---|---|
| YOLOv8n (baseline) | 11.7 MB | 0.650 | 0.389 | Mobile/Web |
| **YOLOv8n + HPO (Lite)** | **11.7 MB** | **0.758** | **0.441** | **Mobile/Web** |
| **YOLOv8m + HPO (Full)** | **98.8 MB** | **0.904** | **0.512** | **Desktop/Web** |

Dataset: 1,129 bone fracture X-ray images (train 903 / val 112 / test 114)

---

## 🚀 Quick Start

### Web Demo (No install)
Visit **[https://bone-fracture-yolov8.netlify.app](https://bone-fracture-yolov8.netlify.app)** — works in any modern browser globally.

### Local (Python serve)
```bash
git clone https://github.com/zvs808-code/bone-fracture-yolov8
cd bone-fracture-yolov8/frontend
python -m http.server 8080
# → Open http://localhost:8080
```

### Mobile App
```bash
cd app
npm install              # also copies ORT WASM files to www/ort/
# Android
npx cap sync android && npx cap open android
# iOS (macOS only)
npx cap sync ios && npx cap open ios
# Desktop Electron
npm run electron:start
```

---

## 📁 Repository Structure

```
bone-fracture-yolov8/
├── frontend/               ← Web demo (deployed to Netlify)
│   ├── index.html          (YOLOv8 detection UI)
│   ├── app.js              (ONNX inference engine)
│   ├── metadata.json       (model metadata)
│   └── model.onnx          (YOLOv8n Lite, 11.7 MB)
├── app/                    ← Cross-platform mobile/desktop app
│   ├── www/                (shared web assets)
│   ├── android/            (Capacitor Android project)
│   ├── ios/                (Capacitor iOS project)
│   ├── electron/           (Electron desktop)
│   └── tools/              (ONNX export, ORT asset copy)
├── training/               ← Training scripts
│   ├── automl_optuna.py    (Optuna HPO pipeline)
│   ├── requirements.txt
│   └── fracture_yolov8_automl.ipynb
├── netlify.toml            ← Netlify deployment config
└── LICENSE                 (MIT)
```

---

## 🔧 Training

```bash
cd training
pip install -r requirements.txt

# Run Optuna AutoML HPO
python automl_optuna.py --model yolov8m --trials 50 --epochs 30

# Full training with best HPO params
yolo detect train model=yolov8m.pt data=fracture.yaml epochs=72 imgsz=640 \
  lr0=0.0082 momentum=0.937 weight_decay=0.00045
```

---

## 📱 Android APK

Pre-built debug APK available in [Releases](https://github.com/zvs808-code/bone-fracture-yolov8/releases).  
Build from source: see `app/README.md`.

---

## 📄 Paper

> Jang, H. "YOLOv8-Based Automatic Bone Fracture Detection in X-Ray Images via Optuna AutoML Hyperparameter Optimization." *Diagnostics (MDPI)* 2026 (submitted).

---

## License

MIT License — Copyright (c) 2026 Jang Ho

See [LICENSE](LICENSE) for details.
