"""inference_pipeline.py — unified defect inspection: object detection + surface anomaly.

Takes a single product image and runs it through two models in parallel:

  * YOLO (ultralytics)     — object detection: flags missing components / damaged packaging
  * PatchCore (anomalib)   — surface defect anomaly detection via memory-bank comparison

Both models are loaded once in __init__ and reused across calls, so the typical
pattern is to create one DefectInspector per process and call .inspect() in a loop.

Device placement is explicit: pass device="auto" to let the code pick GPU when
available, or "cpu" / "0" (CUDA index) to force a specific device. Both models
are sent to the same device; mixed-device setups are not supported.

Usage
-----
    from src.inference_pipeline import DefectInspector

    inspector = DefectInspector(device="auto")
    result = inspector.inspect("path/to/image.jpg")
    print(result["anomaly_score"], result["detections"])

CLI quick-test::

    python -m src.inference_pipeline path/to/image.jpg --device auto
    python -m src.inference_pipeline path/to/image.jpg --device 0 --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src import config

logger = logging.getLogger("inference_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Device resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device(device: str | None) -> str:
    """Normalise a user-supplied device string to a torch device name.

    Accepted inputs → resolved output
    ----------------------------------
    None / "auto"  → "cuda" if a GPU is available, else "cpu"
    "cpu"          → "cpu"
    "cuda" / "gpu" → "cuda"  (raises if no GPU present)
    "0" / "1" ...  → "cuda:0" / "cuda:1" ... (raises if index out of range)
    "cuda:0" ...   → passed through after validation
    """
    d = (device or "auto").strip().lower()

    if d in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"

    if d == "cpu":
        return "cpu"

    if d in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "A CUDA device was requested but none is available. Use device='cpu'."
            )
        return "cuda"

    if d.isdigit():
        idx = int(d)
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"GPU device {idx} requested but CUDA is not available. Use device='cpu'."
            )
        n = torch.cuda.device_count()
        if idx >= n:
            raise RuntimeError(f"CUDA device {idx} requested but only {n} device(s) found.")
        return f"cuda:{idx}"

    if d.startswith("cuda:"):
        try:
            idx = int(d.split(":")[1])
        except ValueError:
            raise ValueError(f"Unrecognised device string '{device}'.") from None
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        if idx >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {idx} not found.")
        return d

    raise ValueError(
        f"Unrecognised device '{device}'. Valid options: auto | cpu | cuda | <int>."
    )


def _anomalib_device_str(torch_device: str) -> str:
    """Convert a torch device string to anomalib's TorchInferencer format.

    anomalib 2.5 TorchInferencer accepts only "auto" | "cpu" | "cuda" | "xpu".
    Specific CUDA indices (e.g. "cuda:0") must be collapsed to just "cuda".
    """
    if torch_device.startswith("cuda"):
        return "cuda"
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Result schema
# ─────────────────────────────────────────────────────────────────────────────

def _empty_result(device: str) -> dict[str, Any]:
    return {
        "device": device,
        # YOLO detections — list of {label, confidence, box: [x1,y1,x2,y2]}
        "detections": [],
        # PatchCore outputs
        "anomaly_score": None,   # float — higher means more anomalous
        "anomaly_map": None,     # np.ndarray (H, W) float32 — pixel-level heatmap
        "is_anomalous": None,    # bool — True if score exceeds threshold
        # Any non-fatal per-model errors are collected here instead of raising
        "errors": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# DefectInspector
# ─────────────────────────────────────────────────────────────────────────────

class DefectInspector:
    """Unified defect inspector: YOLO object detection + PatchCore anomaly detection.

    Parameters
    ----------
    anomaly_run_dir : path-like, optional
        Directory of a trained PatchCore run containing ``weights/torch/model.pt``.
        Defaults to the most-recently-modified run under ``models/anomaly/``.
    yolo_weights : path-like, optional
        Path to YOLO weights (.pt). Defaults to ``yolov8n.pt`` in the project root.
    device : str, optional
        Target device. One of: "auto" | "cpu" | "cuda" | "<int gpu index>".
        Defaults to "auto" — uses GPU when available, CPU otherwise.
    anomaly_threshold : float, optional
        Override the image-level score threshold for the is_anomalous flag.
        When None, uses the threshold baked into the exported model by anomalib.
    """

    def __init__(
        self,
        anomaly_run_dir: str | Path | None = None,
        yolo_weights: str | Path | None = None,
        device: str | None = "auto",
        anomaly_threshold: float | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self._anomaly_threshold = anomaly_threshold
        logger.info("DefectInspector initialising on device: %s", self.device)

        self._yolo = self._load_yolo(yolo_weights)
        self._anomaly_inferencer = self._load_patchcore(anomaly_run_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # Loaders
    # ─────────────────────────────────────────────────────────────────────────

    def _load_yolo(self, weights_path: str | Path | None):
        from ultralytics import YOLO

        path = Path(weights_path) if weights_path else config.PROJECT_ROOT / "yolov8n.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {path}. "
                "Place yolov8n.pt in the project root or pass yolo_weights=<path>."
            )
        logger.info("Loading YOLO weights from %s", path)
        return YOLO(str(path))

    def _load_patchcore(self, run_dir: str | Path | None):
        from anomalib.deploy import TorchInferencer

        if run_dir is None:
            run_dir = _find_latest_anomaly_run()
        run_dir = Path(run_dir)

        model_pt = run_dir / "weights" / "torch" / "model.pt"
        if not model_pt.exists():
            raise FileNotFoundError(
                f"PatchCore TorchScript model not found at {model_pt}.\n"
                "Train it first:  python -m src.train_anomaly --category <cat>"
            )

        logger.info("Loading PatchCore from %s", model_pt)
        # anomalib 2.5 blocks pickle-based TorchScript loading by default.
        # Our model.pt was exported by our own train_anomaly.py — it is trusted.
        # We set the flag in a narrow scope rather than globally.
        import os
        os.environ.setdefault("TRUST_REMOTE_CODE", "1")
        return TorchInferencer(
            path=str(model_pt),
            device=_anomalib_device_str(self.device),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def inspect(self, image: str | Path | np.ndarray | Image.Image) -> dict[str, Any]:
        """Run the full defect inspection pipeline on a single image.

        Parameters
        ----------
        image : str | Path | np.ndarray | PIL.Image
            Input image. File paths are loaded automatically. NumPy arrays must be
            HWC uint8 RGB (or float32 [0, 1] — scaled automatically).

        Returns
        -------
        dict with keys:
            device        : str            — which device the models ran on
            detections    : list[dict]     — YOLO hits: label / confidence / box
            anomaly_score : float          — image-level score (higher = more anomalous)
            anomaly_map   : np.ndarray     — pixel heatmap, shape (H, W), float32
            is_anomalous  : bool           — True if score exceeds the threshold
            errors        : list[str]      — non-fatal per-model error messages, if any
        """
        result = _empty_result(self.device)
        pil_image = _to_pil(image)

        # Each model runs independently — a failure in one doesn't abort the other.
        try:
            result["detections"] = self._run_yolo(pil_image)
        except Exception as exc:
            msg = f"YOLO detection failed: {exc}"
            logger.warning(msg)
            result["errors"].append(msg)

        try:
            score, amap, is_anomalous = self._run_patchcore(pil_image)
            result["anomaly_score"] = score
            result["anomaly_map"] = amap
            result["is_anomalous"] = is_anomalous
        except Exception as exc:
            msg = f"Anomaly detection failed: {exc}"
            logger.warning(msg)
            result["errors"].append(msg)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Model runners
    # ─────────────────────────────────────────────────────────────────────────

    def _run_yolo(self, image: Image.Image) -> list[dict]:
        """Return a list of YOLO detection dicts (label, confidence, box)."""
        # ultralytics expects a CUDA index string ("0") or "cpu", not "cuda:0"
        yolo_device = self.device.replace("cuda:", "") if "cuda" in self.device else "cpu"
        raw_results = self._yolo(image, device=yolo_device, verbose=False)

        detections: list[dict] = []
        for r in raw_results:
            if r.boxes is None:
                continue
            names = r.names
            for box in r.boxes:
                detections.append({
                    "label": names[int(box.cls)],
                    "confidence": round(float(box.conf), 4),
                    "box": [round(v, 1) for v in box.xyxy[0].tolist()],  # [x1,y1,x2,y2]
                })
        return detections

    def _run_patchcore(
        self, image: Image.Image
    ) -> tuple[float, np.ndarray, bool]:
        """Return (image_score, pixel_heatmap, is_anomalous) from PatchCore."""
        # TorchInferencer.predict accepts PIL Images directly (anomalib 2.5+)
        batch = self._anomaly_inferencer.predict(image=image)

        score = float(batch.pred_score)

        # anomaly_map is a tensor of shape (1, H, W) or (H, W); normalise to (H, W)
        amap: np.ndarray = batch.anomaly_map.squeeze().cpu().numpy().astype(np.float32)

        if self._anomaly_threshold is not None:
            is_anomalous = score >= self._anomaly_threshold
        else:
            # pred_label: 1 = anomalous, 0 = normal (set by anomalib during export)
            is_anomalous = bool(int(batch.pred_label))

        return score, amap, is_anomalous


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest_anomaly_run() -> Path:
    """Return the most-recently-modified run dir that has a model.pt inside it."""
    runs = [
        d for d in config.ANOMALY_WEIGHTS_DIR.iterdir()
        if d.is_dir() and (d / "weights" / "torch" / "model.pt").exists()
    ]
    if not runs:
        raise FileNotFoundError(
            f"No trained anomaly runs found under {config.ANOMALY_WEIGHTS_DIR}. "
            "Run `python -m src.train_anomaly` first."
        )
    latest = max(runs, key=lambda d: d.stat().st_mtime)
    logger.info("Auto-selected anomaly run: %s", latest.name)
    return latest


def _to_pil(image: str | Path | np.ndarray | Image.Image) -> Image.Image:
    """Convert any supported input type to an RGB PIL Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        return Image.fromarray(image).convert("RGB")
    raise TypeError(
        f"Unsupported image type '{type(image).__name__}'. "
        "Pass a file path, PIL Image, or HWC uint8 numpy array."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the DefectInspector on a single image and print results."
    )
    parser.add_argument("image", help="Path to the image to inspect")
    parser.add_argument(
        "--device", default="auto",
        help="auto | cpu | cuda | <int gpu index>  (default: auto)",
    )
    parser.add_argument(
        "--anomaly-run-dir", default=None,
        help="Path to a specific PatchCore run dir (default: auto-detect latest)",
    )
    parser.add_argument("--yolo-weights", default=None, help="Path to YOLO .pt weights")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override anomaly score threshold for the is_anomalous flag",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    inspector = DefectInspector(
        anomaly_run_dir=args.anomaly_run_dir,
        yolo_weights=args.yolo_weights,
        device=args.device,
        anomaly_threshold=args.threshold,
    )
    result = inspector.inspect(args.image)

    # Swap the heatmap array for its shape so the JSON stays readable
    printable = {
        k: (list(v.shape) if isinstance(v, np.ndarray) else v)
        for k, v in result.items()
    }
    print(json.dumps(printable, indent=2))
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
