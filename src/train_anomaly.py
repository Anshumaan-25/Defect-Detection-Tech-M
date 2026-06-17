"""train_anomaly.py — train the surface-defect anomaly detector (PatchCore / anomalib).

This is the training entry point for the "surface defects" requirement. It uses
Intel's **anomalib** library with the **PatchCore** algorithm on the MVTec AD
dataset that ``src/dataset_loader.py`` downloads + structures.

Why PatchCore + ResNet18
------------------------
PatchCore is a memory-bank anomaly detector: it extracts features from *normal*
(defect-free) training images with a pretrained CNN backbone, stores a coreset
of them, and at test time flags patches whose nearest stored feature is far
away. It "trains" in a single pass (one epoch) — no gradient descent — so it is
fast and needs no labelled defects.

The defaults here are tuned to run on a **4GB-VRAM Windows GPU** for local
testing, before scaling up on Colab:

  * ``--backbone resnet18``      lightweight vs. anomalib's wide_resnet50_2 default
  * ``--image-size 256``         lower this (e.g. 224 / 128) to cut memory further
  * ``--batch 8``                small feature-extraction batches fit 4GB
  * ``--coreset-ratio 0.1``      smaller coreset = smaller memory bank

Workflow
--------
1. Get + structure the data first (see configs/datasets.yaml for the license note)::

       python -m src.dataset_loader --name mvtec_ad

2. Train one category::

       # local 4GB GPU:
       python -m src.train_anomaly --category bottle --device 0

       # CPU fallback (slow, but works anywhere):
       python -m src.train_anomaly --category bottle --device cpu

       # scale up on Colab (bigger backbone + full image size):
       python -m src.train_anomaly --category bottle --device 0 \
           --backbone wide_resnet50_2 --image-size 256

Results (checkpoint, exported TorchScript model, metrics, visualizations) land
under ``models/anomaly/<run-name>/``.

Colab note
----------
On Colab, install anomalib and run the same two commands in a cell::

    !pip install "anomalib>=1.1.0"
    !python -m src.dataset_loader --name mvtec_ad
    !python -m src.train_anomaly --category bottle --device 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src import config

logger = logging.getLogger("train_anomaly")

# Default PatchCore feature-extraction layers. layer2 + layer3 of a ResNet give
# a good mid-level receptive field while keeping the memory bank small.
DEFAULT_LAYERS = ("layer2", "layer3")


# ──────────────────────────────────────────────────────────────────────────
# Locating the dataset
# ──────────────────────────────────────────────────────────────────────────
def find_mvtec_root(dataset_name: str, category: str) -> Path:
    """Return the MVTec dataset root for anomalib, validating the category layout.

    anomalib's ``MVTecAD`` datamodule wants ``root`` to be the folder that holds
    the category sub-directories, with the category passed separately. We verify
    that ``<root>/<category>/train/good`` exists so failures are caught here with
    a clear message rather than deep inside the data loader.
    """
    root = config.ANOMALY_RAW_DIR / dataset_name
    train_good = root / category / "train" / "good"
    if not train_good.is_dir():
        raise FileNotFoundError(
            f"MVTec category '{category}' not found (expected {train_good}). "
            f"Download + structure it first:\n"
            f"    python -m src.dataset_loader --name {dataset_name}"
        )
    return root


# ──────────────────────────────────────────────────────────────────────────
# Device handling
# ──────────────────────────────────────────────────────────────────────────
def resolve_accelerator(device: str | None) -> tuple[str, object]:
    """Map our ``--device`` flag onto anomalib/Lightning (accelerator, devices).

    Accepted values::

        auto         -> ("auto", 1)   # let Lightning pick (GPU if present)
        cpu          -> ("cpu", 1)
        mps          -> ("mps", 1)    # Apple Silicon
        gpu          -> ("gpu", 1)
        0  / "0,1"   -> ("gpu", [0]) / ("gpu", [0, 1])   # specific CUDA index(es)
    """
    d = (device or "auto").strip().lower()
    if d in ("", "auto"):
        return "auto", 1
    if d in ("cpu", "mps", "gpu"):
        return d, 1
    try:
        ids = [int(x) for x in d.split(",") if x != ""]
    except ValueError:
        raise ValueError(
            f"Unrecognized --device '{device}'. Use auto | cpu | mps | gpu | <int>."
        ) from None
    if not ids:
        raise ValueError(f"Unrecognized --device '{device}'.")
    return "gpu", ids


# ──────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────
def train(
    dataset_name: str = "mvtec_ad",
    category: str = "bottle",
    *,
    backbone: str = "resnet18",
    layers: tuple[str, ...] = DEFAULT_LAYERS,
    image_size: int = 256,
    coreset_ratio: float = 0.1,
    num_neighbors: int = 9,
    batch: int = 8,
    num_workers: int = 4,
    device: str | None = None,
    seed: int | None = None,
    run_name: str | None = None,
    do_export: bool = True,
):
    """Fit PatchCore on one MVTec category and save results under models/anomaly/.

    Returns the list of test metrics dicts produced by ``engine.test``.
    """
    # Imported lazily so `--help` / dataset checks don't require the (heavy)
    # anomalib + lightning stack to be installed.
    from anomalib.data import MVTecAD
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    root = find_mvtec_root(dataset_name, category)
    config.ensure_dirs()
    run_name = run_name or f"{dataset_name}_{category}_patchcore"
    accelerator, devices = resolve_accelerator(device)
    out_dir = config.ANOMALY_WEIGHTS_DIR / run_name

    if seed is not None:
        from lightning.pytorch import seed_everything

        seed_everything(seed, workers=True)

    logger.info("Dataset root  : %s  (category=%s)", root, category)
    logger.info("Backbone      : %s  layers=%s  image_size=%d", backbone, list(layers), image_size)
    logger.info("Coreset ratio : %.3f   num_neighbors=%d   batch=%d", coreset_ratio, num_neighbors, batch)
    logger.info("Accelerator   : %s  devices=%s", accelerator, devices)
    logger.info("Output dir    : %s", out_dir)

    datamodule = MVTecAD(
        root=str(root),
        category=category,
        train_batch_size=batch,
        eval_batch_size=batch,
        num_workers=num_workers,
    )

    model = Patchcore(
        backbone=backbone,
        layers=list(layers),
        pre_trained=True,                 # transfer features from ImageNet weights
        coreset_sampling_ratio=coreset_ratio,
        num_neighbors=num_neighbors,
        # Input resizing/normalization lives in the model's pre-processor in
        # anomalib 2.x — this is where image_size is applied.
        pre_processor=Patchcore.configure_pre_processor(
            image_size=(image_size, image_size)
        ),
    )

    # Engine forwards extra kwargs (accelerator/devices) to the Lightning Trainer.
    # PatchCore sets max_epochs=1 itself, so we don't pass it here.
    engine = Engine(
        accelerator=accelerator,
        devices=devices,
        default_root_dir=str(out_dir),
    )

    logger.info("Fitting PatchCore (single-pass memory-bank build)…")
    engine.fit(model=model, datamodule=datamodule)

    logger.info("Evaluating on the test split…")
    metrics = engine.test(model=model, datamodule=datamodule)
    logger.info("Test metrics: %s", metrics)

    if do_export:
        export_path = engine.export(
            model=model,
            export_type=ExportType.TORCH,
            export_root=str(out_dir),
        )
        logger.info("Exported TorchScript model -> %s", export_path)

    logger.info("Done. Artifacts under %s", out_dir)
    return metrics


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a PatchCore surface-defect anomaly detector (anomalib).",
    )
    parser.add_argument(
        "--dataset", default="mvtec_ad",
        help="dataset folder under data/raw/anomaly/ (default: mvtec_ad)",
    )
    parser.add_argument(
        "--category", default="bottle",
        help="MVTec AD category to train, e.g. bottle, cable, hazelnut (default: bottle)",
    )
    parser.add_argument(
        "--backbone", default="resnet18",
        help="CNN feature backbone (default: resnet18; use wide_resnet50_2 on Colab)",
    )
    parser.add_argument(
        "--layers", nargs="+", default=list(DEFAULT_LAYERS),
        help="backbone layers to extract features from (default: layer2 layer3)",
    )
    parser.add_argument(
        "--image-size", type=int, default=256,
        help="square input size; lower to reduce VRAM (default: 256)",
    )
    parser.add_argument(
        "--coreset-ratio", type=float, default=0.1,
        help="PatchCore coreset sampling ratio; lower = smaller memory bank (default: 0.1)",
    )
    parser.add_argument(
        "--num-neighbors", type=int, default=9,
        help="nearest neighbors for the anomaly score (default: 9)",
    )
    parser.add_argument(
        "--batch", type=int, default=8,
        help="train/eval batch size; small fits 4GB VRAM (default: 8)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="dataloader workers (default: 4; lower on Windows if it stalls)",
    )
    parser.add_argument(
        "--device", default=None,
        help="auto | cpu | mps | gpu | <int gpu index, e.g. 0 or 0,1> (default: auto)",
    )
    parser.add_argument("--seed", type=int, default=None, help="optional random seed for reproducibility")
    parser.add_argument("--name", default=None, help="run name (default: <dataset>_<category>_patchcore)")
    parser.add_argument("--no-export", action="store_true", help="skip exporting the TorchScript model")
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
        args.category,
        backbone=args.backbone,
        layers=tuple(args.layers),
        image_size=args.image_size,
        coreset_ratio=args.coreset_ratio,
        num_neighbors=args.num_neighbors,
        batch=args.batch,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        run_name=args.name,
        do_export=not args.no_export,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
