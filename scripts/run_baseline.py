"""Two-stage CV-HTE inference on a local image or video. No temporal tracking."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLOv5_AUTOINSTALL"] = "false"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/run")
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.source.is_file() or args.sample_seconds <= 0:
        parser.error("Source must exist and sample-seconds must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is nonempty; choose a new --output")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    use_cuda = args.device.startswith("cuda")
    if use_cuda:
        torch.cuda.set_device(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    # Register the local YOLOv5 modules used by its original checkpoint.
    sys.path.insert(0, str(ROOT / "external/yolov5"))
    vial = torch.hub.load(str(ROOT / "external/yolov5"), "custom",
                          path=str(ROOT / "models/vial.pt"), source="local",
                          device=args.device, _verbose=False)
    vial.conf = 0.5
    phase = YOLO(str(ROOT / "models/phase.pt"))
    metadata = {"source": str(args.source.resolve()), "device": args.device,
                "phase_names": phase.names, "sample_seconds": args.sample_seconds,
                "note": "vial_index is frame-local, not a persistent track ID"}
    image_input = args.source.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    cap = None if image_input else cv2.VideoCapture(str(args.source))
    if cap is not None and not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) if cap else None
    if cap and (not fps or fps <= 0):
        raise RuntimeError("Video has no usable frame rate")
    metadata["fps"] = fps
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    index = processed = 0
    stride = max(1, round(fps * args.sample_seconds)) if cap else 1
    if use_cuda:
        torch.cuda.synchronize(args.device)
    loop_started = time.perf_counter()
    try:
        with (args.output / "detections.jsonl").open("w", encoding="utf-8") as log:
            while True:
                if cap:
                    ok, frame = cap.read()
                    if not ok:
                        break
                else:
                    frame = cv2.imread(str(args.source))
                    if frame is None:
                        raise RuntimeError("Cannot decode image")
                current = index
                index += 1
                if current % stride:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with torch.inference_mode():
                    detections = vial(rgb).xyxy[0].cpu().tolist()
                record = {"frame": current, "time_seconds": current / fps if fps else None,
                          "vials": []}
                annotated = frame.copy()
                for i, (x1, y1, x2, y2, confidence, _) in enumerate(detections):
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w, int(x2)), min(h, int(y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    result = phase.predict(frame[y1:y2, x1:x2], imgsz=640, conf=0.3,
                                           iou=0.1, device=args.device, verbose=False)[0]
                    entry = {"vial_index": i, "bbox": [x1, y1, x2, y2],
                             "confidence": confidence, "phases": []}
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 150, 0), 2)
                    for box in result.boxes.cpu():
                        cls = int(box.cls.item())
                        score = float(box.conf.item())
                        a, b, c, d = box.xyxy[0].tolist()
                        coords = [int(a+x1), int(b+y1), int(c+x1), int(d+y1)]
                        label = result.names[cls]
                        entry["phases"].append({"class_id": cls, "label": label,
                                                "confidence": score, "bbox": coords})
                        cv2.rectangle(annotated, tuple(coords[:2]), tuple(coords[2:]), (0, 220, 0), 2)
                        cv2.putText(annotated, f"{label} {score:.2f}", tuple(coords[:2]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
                    record["vials"].append(entry)
                log.write(json.dumps(record) + "\n")
                log.flush()
                if not cv2.imwrite(str(args.output / f"frame_{current:08d}.jpg"), annotated):
                    raise RuntimeError("Could not save annotated frame")
                processed += 1
                print(f"frame={current} vials={len(record['vials'])}", flush=True)
                if image_input or (args.max_frames and processed >= args.max_frames):
                    break
    finally:
        if cap:
            cap.release()
    if not processed:
        raise RuntimeError("No frames decoded")
    if use_cuda:
        torch.cuda.synchronize(args.device)
    ended = time.perf_counter()
    metrics = {
        "processed_frames": processed,
        "decoded_frames": index,
        "effective_sample_seconds": stride / fps if fps else None,
        "model_setup_seconds": loop_started - started,
        "processing_seconds": ended - loop_started,
        "total_seconds_after_arguments": ended - started,
        "sampled_frames_per_processing_second": processed / (ended - loop_started),
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(args.device) / 2**20 if use_cuda else None,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(args.device) / 2**20 if use_cuda else None,
        "note": "Processing includes decoding, first-inference warmup, inference and image/JSON writes. CUDA metrics cover this process's PyTorch allocator, not total GPU usage."
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {processed} sampled frames to {args.output}")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
