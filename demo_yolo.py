"""demo_yolo.py -- visualise the trained PCB-defect YOLO detector on an image.

Loads the trained YOLO weights (models/yolo/pcb_defects_yolo/weights/best.pt),
runs detection on a PCB image, draws the predicted boxes + class labels, saves
the annotated image, and opens it. Use this to show the object-detection half of
the pipeline live (separately from the metal-nut anomaly demo in inspect_image.py).

Usage
-----
    # Auto-pick the first PCB test image and annotate it:
    python demo_yolo.py

    # A specific image:
    python demo_yolo.py path/to/pcb.jpg

    # Tune confidence / device / output:
    python demo_yolo.py path/to/pcb.jpg --conf 0.4 --device 0 --out boxes.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "models/yolo/pcb_defects_yolo/weights/best.pt"
PCB_TEST_DIR = PROJECT_ROOT / "data/raw/yolo/pcb_defects/test/images"


def _default_image() -> Path | None:
    """Pick the first PCB test image, if the dataset is present."""
    if PCB_TEST_DIR.is_dir():
        for p in sorted(PCB_TEST_DIR.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                return p
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Annotate an image with the trained PCB YOLO detector.")
    parser.add_argument("image", nargs="?", default=None,
                        help="image to run detection on (default: first PCB test image)")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO .pt weights")
    parser.add_argument("--device", default="auto", help="auto | cpu | 0 (gpu index)")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold (default: 0.25)")
    parser.add_argument("--out", default=None, help="output PNG (default: result_yolo_<name>.png)")
    parser.add_argument("--no-open", action="store_true", help="don't open the result")
    args = parser.parse_args(argv)

    # Resolve the input image.
    image_path = Path(args.image) if args.image else _default_image()
    if image_path is None or not image_path.exists():
        print("[ERROR] No image found. Pass a path, or download the PCB dataset:")
        print("        python -m src.dataset_loader --name pcb_defects")
        return 1

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERROR] Trained weights not found at {weights}.")
        print("        Train first: python -m src.train_yolo --dataset pcb_defects --device 0")
        return 1

    out_path = Path(args.out) if args.out else PROJECT_ROOT / f"result_yolo_{image_path.stem}.png"

    print("\n==================================================")
    print("  PCB Defect Detection -- YOLO Demo")
    print("==================================================")
    print(f"  Image   : {image_path.name}")
    print(f"  Weights : {weights.name}")

    # Lazy import so --help works without the heavy stack.
    from ultralytics import YOLO

    yolo_device = "cpu" if args.device in ("cpu",) else (args.device if args.device != "auto" else None)
    model = YOLO(str(weights))
    results = model(str(image_path), conf=args.conf, device=yolo_device, verbose=False)
    r = results[0]

    # Print detections.
    n = 0 if r.boxes is None else len(r.boxes)
    print(f"  Detections ({n}):")
    if n:
        for box in r.boxes:
            label = r.names[int(box.cls)]
            print(f"    - {label:<16} conf={float(box.conf):.3f}")
    else:
        print("    (none above threshold)")

    # r.plot() returns a BGR annotated image; convert to RGB for PIL.
    annotated_bgr = r.plot()
    Image.fromarray(annotated_bgr[..., ::-1]).save(out_path)
    print(f"  Saved -> {out_path}")
    print("==================================================\n")

    if not args.no_open:
        try:
            os.startfile(str(out_path))
        except (AttributeError, OSError):
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
