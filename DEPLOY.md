# Deployment Guide

## 🌐 Deploy to Netlify (Free)

### Option 1: Netlify Drop (Fastest — no account needed for quick test)
1. Open https://app.netlify.com/drop
2. Drag the `frontend/` folder onto the page
3. Get instant URL (e.g., https://random-name.netlify.app)

### Option 2: GitHub + Netlify (Permanent, with custom URL)
1. Push this repo to GitHub: `https://github.com/zvs808-code/bone-fracture-yolov8`
2. Go to https://app.netlify.com → "Add new site" → "Import from Git"
3. Select your GitHub repo
4. Build settings:
   - Build command: (leave empty)
   - Publish directory: `frontend`
5. Click "Deploy site"
6. Custom domain: Site settings → Domain management → `bone-fracture-yolov8.netlify.app`

The `netlify.toml` in the repo root automatically configures:
- CORS headers required for ONNX Runtime WASM (SharedArrayBuffer)
- Long cache for .onnx and .wasm files
- SPA fallback redirect

## 📦 Push to GitHub
```bash
cd E:\bone-fracture-yolov8-github
git init
git lfs install
git lfs track "*.onnx" "*.wasm" "*.pt" "*.apk"
git add .
git commit -m "Initial release: YOLOv8 bone fracture detector, mAP50=90.4%"
git branch -M main
git remote add origin https://github.com/zvs808-code/bone-fracture-yolov8.git
git push -u origin main
git tag -a v1.0.0 -m "v1.0.0 — YOLOv8m mAP50=0.904"
git push origin v1.0.0
```

## 📱 Build APK
```bash
cd E:\bone-fracture-yolov8-github\app
npm install
npx cap sync android
cd android
./gradlew assembleDebug
# APK: android/app/build/outputs/apk/debug/app-debug.apk
# Rename: 장호_FractureDetector_v1.0.0_debug.apk
```
