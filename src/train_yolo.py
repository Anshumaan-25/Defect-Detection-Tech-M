"""train_yolo.py — train the YOLO detector for missing-components / damaged-packaging.

This is the training entry point for the object-detection half of the project.
The *heavy* training is meant to run on **Google Colab** (free GPU); this script
is written so the exact same command works there and on a laptop. The only
difference is the hardware — paths come from ``src/config.py``, so nothing is
hard-coded to one machine.

Workflow
--------
1. Download the dataset first (writes a Roboflow YOLOv8 export, including a
   ``data.yaml``, under ``data/raw/yolo/<name>/``)::

       python -m src.dataset_loader --name pcb_defects

2. Train against it::

       # laptop smoke test (few epochs, CPU/GPU auto):
       python -m src.train_yolo --dataset pcb_defects --epochs 5

       # full run (typically on Colab with a GPU):
       python -m src.train_yolo --dataset pcb_defects --epochs 100 --batch 16 --device 0

Trained weights + run artifacts land under ``models/yolo/<run-name>/``.

Colab note
----------
On Colab, set the Roboflow key and run the two commands above in a cell, e.g.::

    import os; os.environ["ROBOFLOW_API_KEY"] = "<key>"
    !python -m src.dataset_loader --name pcb_defects
    !python -m src.train_yolo --dataset pcb_defects --epochs 100 --device 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src import config

logger = logging.getLogger("train_yolo")


# ──────────────────────────────────────────────────────────────────────────
# Locating the dataset
# ──────────────────────────────────────────────────────────────────────────
def find_data_yaml(dataset_name: str) -> Path:
    """Return the path to a downloaded dataset's ``data.yaml``.

    Roboflow YOLOv8 exports always include a ``data.yaml`` that lists the
    train/val/test splits and class names — this is what ultralytics consumes.
    It usually sits at the dataset root, but Roboflow sometimes nests it one
    level deep, so we search recursively and take the shallowest match.
    """
    root = config.YOLO_RAW_DIR / dataset_name
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found at {root}. "
            f"Download it first:  python -m src.dataset_loader --name {dataset_name}"
        )

    candidates = sorted(root.rglob("data.yaml"), key=lambda p: len(p.parts))
    if not candidates:
        raise FileNotFoundError(
            f"No data.yaml found under {root}. Is this a YOLO-format export?"
        )
    return candidates[0]


# ──────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────
def train(
    dataset_name: str,
    *,
    model: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    workers: int = 8,
    device: str | None = None,
    run_name: str | None = None,
):
    """Train a YOLO detector on the named dataset and save under models/yolo/.

    Parameters mirror the most-used ultralytics knobs. ``device`` is passed
    straight through: None lets ultralytics auto-pick (GPU if present), "0"
    forces the first CUDA GPU (Colab), "cpu" forces CPU.
    """
    # Imported lazily so `--help` and dataset discovery don't require the (heavy)
    # ultralytics + torch stack to be installed.
    from ultralytics import YOLO

    data_yaml = find_data_yaml(dataset_name)
    config.ensure_dirs()
    run_name = run_name or f"{dataset_name}_yolo"

    logger.info("Dataset config : %s", data_yaml)
    logger.info("Base model     : %s", model)
    logger.info("Epochs / imgsz : %s / %s", epochs, imgsz)
    logger.info("Output dir     : %s", config.YOLO_WEIGHTS_DIR / run_name)

    yolo = YOLO(model)  # pretrained checkpoint -> transfer learning
    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        project=str(config.YOLO_WEIGHTS_DIR),
        name=run_name,
        exist_ok=True,
    )
    logger.info("Training complete. Best weights under %s",
                config.YOLO_WEIGHTS_DIR / run_name / "weights")
    return results


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a YOLO object detector on a downloaded YOLO-format dataset.",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="dataset name as it appears in configs/datasets.yaml (e.g. pcb_defects)",
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="base checkpoint or .yaml (default: yolov8n.pt; use yolov8s/m/l/x for bigger)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="training epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="batch size (default: 16)")
    parser.add_argument(
        "--workers", type=int, default=8,
        help="dataloader workers; lower on RAM-limited machines to avoid OOM (default: 8)",
    )
    parser.add_argument(
        "--device", default=None,
        help="ultralytics device: None=auto, '0' for first GPU, 'cpu' to force CPU",
    )
    parser.add_argument("--name", default=None, help="run name (default: <dataset>_yolo)")
    parser.add_argument("--verbose", "-v", action="store_true", help="enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    train(
        args.dataset,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        run_name=args.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
