"""
YOLOv8 (.pt) → ONNX 변환 스크립트.

본인이 학습한 YOLOv8 best.pt 를 앱에 탑재할 때 실행.

사용법:
  cd 장호_app(android, or IOS)/tools
  python export_onnx.py --weights path/to/best.pt --size 640

출력:
  ../www/model.onnx       (ONNX 모델)
  ../www/metadata.json    (메타데이터)
"""
import argparse, json, pathlib, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="YOLOv8 .pt 가중치 경로")
    ap.add_argument("--size", type=int, default=640, help="입력 이미지 크기 (기본 640)")
    ap.add_argument("--out-dir", type=str, default=None, help="출력 디렉토리 (기본: ../www)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("❌ ultralytics 미설치 — pip install ultralytics")

    here = pathlib.Path(__file__).resolve().parent
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else (here.parent / "www")
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = pathlib.Path(args.weights).resolve()
    print(f"📂 weights : {weights}")
    print(f"📐 img_size: {args.size}")

    model = YOLO(str(weights))

    # ONNX export
    onnx_path = model.export(
        format="onnx",
        imgsz=args.size,
        opset=17,
        simplify=True,
        dynamic=False,
    )
    onnx_src = pathlib.Path(onnx_path)
    onnx_dst = out_dir / "model.onnx"
    import shutil
    shutil.copy2(onnx_src, onnx_dst)
    size_mb = onnx_dst.stat().st_size / 1e6
    print(f"✅ model.onnx : {onnx_dst}  ({size_mb:.1f} MB)")

    # metadata.json
    class_names = list(model.names.values()) if hasattr(model, 'names') else ["fracture"]
    meta = {
        "model_name": f"YOLOv8-{args.size}",
        "model_label": f"YOLOv8 Fracture Detector ({args.size}px)",
        "task": "detection",
        "img_size": args.size,
        "class_names": class_names,
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "input_name": "images",
        "output_name": "output0",
        "normalize_mean": [0.0, 0.0, 0.0],
        "normalize_std": [1.0, 1.0, 1.0],
        "source_weights": str(weights),
        "note": "divide by 255, letterbox to img_size, NCHW float32",
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ metadata.json : {meta_path}")

if __name__ == "__main__":
    main()
