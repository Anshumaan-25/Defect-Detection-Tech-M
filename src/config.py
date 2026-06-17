"""Central configuration: filesystem layout + dataset registry.

Every script imports paths from here instead of hard-coding strings, so the
project behaves identically on a laptop and on Google Colab. The only thing
that changes between environments is PROJECT_ROOT, which is derived
automatically from this file's location.

Secrets (Roboflow API key, etc.) are read from environment variables / a local
.env file and are never written to disk in this repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load variables from a local .env (if present) into the environment.
# .env is git-ignored — keep your API keys there, never in source.
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────
# Filesystem layout
# ──────────────────────────────────────────────────────────────────────────
# config.py lives in <root>/src/, so the project root is one level up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"            # untouched downloads, per modality
PROCESSED_DIR: Path = DATA_DIR / "processed"  # normalized / ready-to-train

# Per-modality raw download targets.
YOLO_RAW_DIR: Path = RAW_DIR / "yolo"          # detection datasets (Roboflow)
ANOMALY_RAW_DIR: Path = RAW_DIR / "anomaly"    # MVTec AD surface defects
OCR_RAW_DIR: Path = RAW_DIR / "ocr"            # label / text images

MODELS_DIR: Path = PROJECT_ROOT / "models"
YOLO_WEIGHTS_DIR: Path = MODELS_DIR / "yolo"
ANOMALY_WEIGHTS_DIR: Path = MODELS_DIR / "anomaly"

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DATASET_REGISTRY_PATH: Path = CONFIGS_DIR / "datasets.yaml"

# Directories that must exist before any download / training run.
_REQUIRED_DIRS = (
    YOLO_RAW_DIR,
    ANOMALY_RAW_DIR,
    OCR_RAW_DIR,
    PROCESSED_DIR,
    YOLO_WEIGHTS_DIR,
    ANOMALY_WEIGHTS_DIR,
)


def ensure_dirs() -> None:
    """Create every project directory that downloads/training depend on."""
    for directory in _REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# Secrets
# ──────────────────────────────────────────────────────────────────────────
def get_roboflow_api_key() -> str | None:
    """Return the Roboflow API key from the environment, or None if unset."""
    return os.getenv("ROBOFLOW_API_KEY")


# ──────────────────────────────────────────────────────────────────────────
# Dataset registry
# ──────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DatasetSpec:
    """A single source dataset described in configs/datasets.yaml.

    Attributes
    ----------
    name      : Short identifier used for the on-disk folder name.
    modality  : One of {"yolo", "anomaly", "ocr"} — picks the loader path.
    source    : Where to fetch from, e.g. "roboflow" or "url".
    enabled   : Skip the entry without deleting it when False.
    params    : Source-specific arguments (workspace/project/version, url, ...).
    """

    name: str
    modality: str
    source: str
    enabled: bool = True
    params: dict | None = None

    def target_dir(self) -> Path:
        """Where this dataset's raw files should land, based on modality."""
        roots = {
            "yolo": YOLO_RAW_DIR,
            "anomaly": ANOMALY_RAW_DIR,
            "ocr": OCR_RAW_DIR,
        }
        if self.modality not in roots:
            raise ValueError(
                f"Unknown modality '{self.modality}' for dataset '{self.name}'. "
                f"Expected one of {sorted(roots)}."
            )
        return roots[self.modality] / self.name


def load_dataset_registry(path: Path = DATASET_REGISTRY_PATH) -> list[DatasetSpec]:
    """Parse configs/datasets.yaml into a list of DatasetSpec objects.

    Returns only entries marked enabled. Raises FileNotFoundError with a clear
    message if the registry is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset registry not found at {path}. "
            "Create it (see configs/datasets.yaml.example) before loading data."
        )

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    specs: list[DatasetSpec] = []
    for entry in raw.get("datasets", []):
        spec = DatasetSpec(
            name=entry["name"],
            modality=entry["modality"],
            source=entry["source"],
            enabled=entry.get("enabled", True),
            params=entry.get("params", {}),
        )
        if spec.enabled:
            specs.append(spec)
    return specs
