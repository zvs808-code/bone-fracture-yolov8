# 장호_app(android, or IOS) — 빌드 및 실행 가이드

YOLOv8n 기반 온디바이스 골절 탐지 앱 (Capacitor Android/iOS + Electron 데스크탑).

---

## 구조

```
장호_app(android, or IOS)/
├── www/                  ← 공통 웹 진입점
│   ├── index.html        (앱 UI)
│   ├── app.js            (YOLOv8 ONNX 추론 로직)
│   ├── model.onnx        (YOLOv8n Lite, 11.7 MB, mAP50=75.8%)
│   └── metadata.json     (모델 메타데이터)
├── electron/             ← 데스크탑 (macOS/Windows)
├── tools/                ← ONNX 변환 스크립트
├── package.json
└── capacitor.config.json
```

---

## 🖥️ 데스크탑 (Electron) — 가장 빠른 실행 방법

```bash
# 1. Node.js 18+ 설치 후
npm install
npm run electron:start
```

또는 CDN 연결 상태에서 단순 브라우저 열기:
```bash
npm run serve
# → http://localhost:3000 으로 접속
```

---

## 🤖 Android 빌드

### 사전 준비
- Node.js 18+, Android Studio, JDK 17+
- `ANDROID_HOME` 환경변수 설정

```bash
npm install
npm install -g @capacitor/cli

# ORT WASM 파일 복사
node tools/copy_ort_assets.js

# Android 프로젝트 동기화
npx cap sync android

# Android Studio에서 빌드
npx cap open android
# → Build > Generate Signed Bundle/APK
```

### APK 직접 빌드 (커맨드라인)
```bash
cd android
./gradlew assembleDebug
# → android/app/build/outputs/apk/debug/app-debug.apk
```

---

## 🍏 iOS 빌드 (macOS 필요)

```bash
npm install
npm install -g @capacitor/cli
sudo gem install cocoapods

npx cap sync ios
cd ios/App
pod install

npx cap open ios
# → Xcode에서 본인 Apple ID 서명 후 Run
```

---

## 모델 교체

더 높은 mAP50의 YOLOv8m 모델(98.8 MB)로 교체하려면:

```bash
cd tools
python export_onnx.py \
  --weights "G:/AI-WEB/training/model.onnx" \
  --size 640
```

또는 기존 ONNX 파일 직접 복사:
```
G:\AI-WEB\frontend\model.onnx  →  www\model.onnx
```
그 후 `www/metadata.json` 의 `full_model_map50`, `lite_model_map50` 수치를 업데이트.

---

## 앱 동작 원리

1. 이미지 선택 → letterbox 640×640 → Float32 / 255 → NCHW
2. YOLOv8n ONNX 추론 → output0 [1, 5, 8400]
3. 신뢰도 필터링 → NMS (IoU 0.45)
4. 원본 좌표 역변환 → Canvas에 바운딩 박스 오버레이
5. 결과: 탐지 개수, 최고 신뢰도, 각 박스 좌표·점수 표시
