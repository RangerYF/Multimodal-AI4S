"""Fixed-threshold detection evaluation; no training or threshold tuning.

oracle_phase_crop uses ground-truth phase-box union and is diagnostic only.
This is not a reproduction of the paper's mAP or user-study scoring protocol.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLOv5_AUTOINSTALL"] = "false"


def iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gt, predictions, threshold=.5):
    """Class-aware, score-ordered, one-to-one IoU matching."""
    matched = set()
    counts = np.zeros((5, 3), dtype=int)  # TP FP FN
    for pred in sorted(predictions, key=lambda p: -p["confidence"]):
        candidates = [(iou(pred["bbox"], g["bbox"]), i) for i, g in enumerate(gt)
                      if i not in matched and g["class_id"] == pred["class_id"]]
        overlap, index = max(candidates, default=(0, -1))
        if index >= 0 and overlap >= threshold:
            matched.add(index)
            counts[pred["class_id"], 0] += 1
        else:
            counts[pred["class_id"], 1] += 1
    for i, g in enumerate(gt):
        if i not in matched:
            counts[g["class_id"], 2] += 1
    return counts


def scores(counts):
    tp, fp, fn = map(int, counts)
    return dict(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else 0,
                recall=tp/(tp+fn) if tp+fn else 0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)


def draw(frame, items, color):
    canvas = frame.copy()
    for item in items:
        a, b, c, d = map(int, item["bbox"])
        cv2.rectangle(canvas, (a, b), (c, d), color, 2)
        label = item["label"] + (f" {item['confidence']:.2f}" if "confidence" in item else "")
        cv2.putText(canvas, label, (a, max(15, b-4)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
    return canvas


def panel(frame, items, region, color, title):
    x1, y1, x2, y2 = region
    crop = cv2.resize(frame[y1:y2, x1:x2], (320, 600))
    transformed = []
    for item in items:
        a, b, c, d = item['bbox']
        transformed.append({**item, 'bbox': [(a-x1)*320/(x2-x1), (b-y1)*600/(y2-y1),
                                            (c-x1)*320/(x2-x1), (d-y1)*600/(y2-y1)]})
    canvas = np.full((640, 320, 3), 255, dtype=np.uint8)
    canvas[40:] = draw(crop, transformed, color)
    cv2.putText(canvas, title, (6,25), cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Root with data.yaml and split/images, split/labels")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    images = sorted((args.dataset / args.split / "images").glob("*.jpg"))
    assert images, "No images"
    names = yaml.safe_load((args.dataset / "data.yaml").read_text())["names"]
    assert names == ["Empty", "Hetero", "Homo", "Residue", "Solid"], names
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, str(ROOT / "external/yolov5"))
    vial = torch.hub.load(str(ROOT / "external/yolov5"), "custom", path=str(ROOT / "models/vial.pt"),
                          source="local", device=device, _verbose=False)
    vial.conf = .5
    phase = YOLO(str(ROOT / "models/phase.pt"))
    assert list(phase.names.values()) == names
    modes = ["automatic", "oracle_phase_crop"]
    totals = {m: np.zeros((5, 3), dtype=int) for m in modes}
    presence = {m: np.zeros((5, 3), dtype=int) for m in modes}
    rows = []
    start = time.perf_counter()
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as log:
        for index, path in enumerate(images):
            original = cv2.imread(str(path))
            assert original is not None, str(path)
            # Match upstream spatial resize, using standard BGR for Ultralytics phase inference.
            frame = cv2.resize(original, (1920, 1080))
            h, w = frame.shape[:2]
            gt = []
            label_path = path.parent.parent / "labels" / (path.stem + ".txt")
            for line in label_path.read_text().splitlines():
                cls, cx, cy, bw, bh = map(float, line.split())
                cls = int(cls)
                gt.append(dict(class_id=cls, label=names[cls], bbox=[(cx-bw/2)*w, (cy-bh/2)*h,
                                                                               (cx+bw/2)*w, (cy+bh/2)*h]))
            assert gt, "Oracle diagnostic requires at least one phase box"
            union = [max(0, int(min(g['bbox'][0] for g in gt))-1),
                     max(0, int(min(g['bbox'][1] for g in gt))-1),
                     min(w, int(max(g['bbox'][2] for g in gt))+1),
                     min(h, int(max(g['bbox'][3] for g in gt))+1)]
            with torch.inference_mode():
                boxes = vial(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).xyxy[0].cpu().tolist()
            rois = {"automatic": [b[:4] for b in boxes], "oracle_phase_crop": [union]}
            record = {"image": path.name, "gt": gt, "vial_boxes": boxes, "predictions": {}}
            for mode in modes:
                preds = []
                for box in rois[mode]:
                    a, b, c, d = [int(x) for x in box]
                    a, b, c, d = max(0, a), max(0, b), min(w, c), min(h, d)
                    if c <= a or d <= b:
                        continue
                    result = phase.predict(frame[b:d, a:c], imgsz=640, conf=.3, iou=.1,
                                           device=device, verbose=False)[0]
                    for pred in result.boxes.cpu():
                        cls = int(pred.cls.item())
                        x1, y1, x2, y2 = pred.xyxy[0].tolist()
                        preds.append(dict(class_id=cls, label=names[cls], confidence=float(pred.conf.item()),
                                          bbox=[x1+a, y1+b, x2+a, y2+b]))
                counts = match(gt, preds)
                totals[mode] += counts
                for cls in range(5):
                    g, p = any(x['class_id'] == cls for x in gt), any(x['class_id'] == cls for x in preds)
                    presence[mode][cls] += [int(g and p), int(p and not g), int(g and not p)]
                record["predictions"][mode] = preds
                row = {"image": path.name, "mode": mode, "gt_boxes": len(gt), "pred_boxes": len(preds),
                       "vial_boxes": len(boxes), **scores(counts.sum(axis=0))}
                rows.append(row)
                if index % 40 == 0 or (mode == "automatic" and row["f1"] < .5 and index < 30):
                    # Compare GT and predictions in the same context; use GT region for display only.
                    x1, y1, x2, y2 = union
                    x1, y1, x2, y2 = max(0,x1-20), max(0,y1-20), min(w,x2+20), min(h,y2+20)
                    region = (x1,y1,x2,y2)
                    comparison = np.hstack([panel(frame, gt, region, (0,180,0), "Published labels"),
                                            panel(frame, preds, region, (0,140,255), mode)])
                    cv2.imwrite(str(args.output / f"example_{index:03d}_{mode}.jpg"), comparison)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if index % 20 == 0:
                print(f"Processed {index+1}/{len(images)}", flush=True)
    report = {"images": len(images), "dataset": str(args.dataset.resolve()), "split": args.split,
              "weights_sha256": {name: hashlib.sha256((ROOT / "models" / name).read_bytes()).hexdigest()
                                  for name in ["vial.pt", "phase.pt"]},
              "device": device, "elapsed_processing_seconds": time.perf_counter()-start,
              "protocol": {"vial_conf": .5, "phase_conf": .3, "phase_nms_iou": .1, "matching_iou": .5,
                           "image_resize": [1920,1080], "phase_input": "BGR", "phase_imgsz": 640,
                           "oracle_note": "Ground-truth phase union crop; unavailable at deployment; not a true annotated vial box",
                           "scope_note": "All phase predictions evaluated against published phase labels. Not paper mAP reproduction."},
              "modes": {m: {"micro": scores(totals[m].sum(axis=0)),
                            "per_class": {names[c]: scores(totals[m][c]) for c in range(5)},
                            "image_presence_per_class": {names[c]: scores(presence[m][c]) for c in range(5)}} for m in modes}}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
