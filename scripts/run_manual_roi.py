"""Evaluate the phase model on visually annotated ROIs, bypassing vial detection."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def render_roi(crop, phases, origin):
    """Enlarge before drawing text so small crops have readable labels."""
    h, w = crop.shape[:2]
    canvas = np.full((280, 320, 3), 255, dtype=np.uint8)
    canvas[40:] = cv2.resize(crop, (320, 240))
    title = "; ".join(f"{p['label']} {p['confidence']:.2f}" for p in phases) or "No detection"
    cv2.putText(canvas, title, (6, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1)
    for p in phases:
        a, b, c, d = p["bbox"]
        pts = (int((a-origin[0])*320/w), int((b-origin[1])*240/h)+40,
               int((c-origin[0])*320/w), int((d-origin[1])*240/h)+40)
        cv2.rectangle(canvas, pts[:2], pts[2:], (0, 220, 0), 2)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.annotations.read_text(encoding="utf-8"))
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(ROOT / "models/phase.pt"))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cap = cv2.VideoCapture(str(ROOT / manifest["source"]))
    if not cap.isOpened():
        raise RuntimeError("Cannot open source")
    rows = []
    try:
        with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as log:
            for entry in manifest["entries"]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, entry["frame"])
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot read frame {entry['frame']}")
                x1, y1, x2, y2 = entry["bbox"]
                h, w = frame.shape[:2]
                if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                    raise ValueError("ROI outside source frame")
                crop = frame[y1:y2, x1:x2].copy()
                pred = model.predict(crop, imgsz=640, conf=0.3, iou=0.1,
                                     device=device, verbose=False)[0]
                phases = []
                for b in pred.boxes.cpu():
                    a, c, d, e = b.xyxy[0].tolist()
                    phases.append({"label": pred.names[int(b.cls.item())],
                                   "confidence": float(b.conf.item()),
                                   "bbox": [a+x1, c+y1, d+x1, e+y1]})
                result = {**entry, "roi_source": "assistant_visual_annotation",
                          "phases": phases}
                log.write(json.dumps(result, ensure_ascii=False) + "\n")
                stem = f"frame_{entry['frame']:08d}_{entry['object_id']}"
                cv2.imwrite(str(args.output / (stem + "_input.png")), crop)
                # Enlarged for reading only; does not add original image detail.
                cv2.imwrite(str(args.output / (stem + "_prediction.jpg")),
                            render_roi(crop, phases, (x1, y1)))
                rows.append({"time_seconds": entry["time_seconds"], "object_id": entry["object_id"],
                             "labels": "; ".join(f"{p['label']} {p['confidence']:.3f}" for p in phases) or "no_detection",
                             "scorable": entry["scorable"], "phase_gt": entry["phase_gt"],
                             "occlusion": entry["occlusion"]})
    finally:
        cap.release()
    with (args.output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "annotations.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "metadata.json").write_text(json.dumps({"device": device, "phase_names": model.names,
        "conf": 0.3, "iou": 0.1, "imgsz": 640, "vial_detector_used": False,
        "scorable_rois": sum(r["scorable"] for r in rows), "accuracy": None}, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
