"""dataset_loader.py — download & organize open-source defect datasets.

This is the single entry point for getting data onto disk. It reads the
dataset registry (configs/datasets.yaml), then dispatches each entry to the
right downloader based on its ``source`` field:

    roboflow  ->  YOLO detection datasets (PCB, bottle inspection, ...)
    url       ->  direct archive download + extract (e.g. MVTec AD mirror)

Everything is config-driven so adding a dataset means editing one YAML entry,
not touching this file. Designed to run identically on a laptop and on Colab.

Usage
-----
    # Download every enabled dataset in the registry:
    python -m src.dataset_loader --all

    # Download a single dataset by name:
    python -m src.dataset_loader --name pcb_defects

    # List what the registry contains without downloading:
    python -m src.dataset_loader --list

Set your Roboflow key first (locally via a .env file, on Colab via os.environ):
    ROBOFLOW_API_KEY=xxxxxxxxxxxx
"""

from __future__ import annotations

import argparse
import logging
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from src import config
from src.config import DatasetSpec

logger = logging.getLogger("dataset_loader")


# ──────────────────────────────────────────────────────────────────────────
# Source-specific downloaders
# ──────────────────────────────────────────────────────────────────────────
def _download_from_roboflow(spec: DatasetSpec) -> Path:
    """Pull a YOLO-format dataset from Roboflow Universe.

    Expected params in the registry entry::

        params:
          workspace: "<workspace-slug>"
          project:   "<project-slug>"
          version:   1          # an integer, or the string "latest"
          format:    "yolov8"   # optional, defaults to yolov8

    Returns the directory the dataset was written to.
    """
    api_key = config.get_roboflow_api_key()
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Add it to a local .env file or, on "
            "Colab, run: import os; os.environ['ROBOFLOW_API_KEY'] = '<key>'."
        )

    # Imported lazily so the rest of the loader (e.g. URL downloads) works even
    # if the roboflow package isn't installed yet.
    from roboflow import Roboflow

    params = spec.params or {}
    workspace = params["workspace"]
    project_slug = params["project"]
    fmt = params.get("format", "yolov8")

    target = spec.target_dir()

    # If a complete export is already on disk, don't re-download. We detect this
    # by the presence of data.yaml rather than by the folder merely existing —
    # the Roboflow SDK's own overwrite=False check treats any existing directory
    # as "done" and silently skips, which left empty folders behind.
    if (target / "data.yaml").exists():
        logger.info("Dataset already present (data.yaml found), skipping: %s", target)
        return target

    target.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_slug)

    # version may be a concrete integer or the string "latest" (resolved here).
    version = _resolve_roboflow_version(project, params.get("version", "latest"))

    logger.info(
        "Roboflow download: %s/%s v%s -> %s", workspace, project_slug, version, target
    )
    dataset = project.version(version).download(
        fmt, location=str(target), overwrite=True
    )
    logger.info("Roboflow dataset ready at %s", dataset.location)
    return Path(dataset.location)


def _resolve_roboflow_version(project, version_param) -> int:
    """Turn a registry ``version`` value into a concrete Roboflow version number.

    Accepts an int (or numeric string) verbatim. The string ``"latest"`` is
    resolved by querying the project's versions and returning the highest one,
    so the registry never has to be bumped when a new dataset version is added.
    """
    if not (isinstance(version_param, str) and version_param.lower() == "latest"):
        return int(version_param)

    versions = project.versions()
    if not versions:
        raise RuntimeError(
            f"Roboflow project '{project.id}' has no versions to download."
        )
    # A Version's `.version` looks like "workspace/project/<n>"; take the last
    # path segment as the numeric id and pick the maximum.
    numbers = [int(str(v.version).rstrip("/").split("/")[-1]) for v in versions]
    latest = max(numbers)
    logger.info("Resolved version 'latest' -> v%s", latest)
    return latest


def _download_from_url(spec: DatasetSpec) -> Path:
    """Stream a (zip/tar) archive from a direct URL and extract it.

    Expected params in the registry entry::

        params:
          url: "https://.../archive.zip"

    Used for datasets that aren't on Roboflow — e.g. an MVTec AD mirror.
    Returns the directory the archive was extracted into.
    """
    params = spec.params or {}
    url = params["url"]

    target = spec.target_dir()
    target.mkdir(parents=True, exist_ok=True)

    archive_path = target / Path(url.split("?")[0]).name
    if archive_path.exists():
        logger.info("Archive already present, skipping download: %s", archive_path)
    else:
        _stream_download(url, archive_path)

    _extract_archive(archive_path, target)
    return target


def _stream_download(url: str, dest: Path) -> None:
    """Download a URL to ``dest`` with a tqdm progress bar (streamed)."""
    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MiB
                fh.write(chunk)
                bar.update(len(chunk))
        tmp.rename(dest)  # atomic: only a complete file gets the final name


def _extract_archive(archive: Path, target: Path) -> None:
    """Extract a .zip / .tar.* archive into ``target`` (no-op for others)."""
    if zipfile.is_zipfile(archive):
        logger.info("Extracting zip %s", archive.name)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    elif archive.suffixes and archive.suffixes[-1] in {".tar", ".gz", ".xz", ".tgz"}:
        logger.info("Extracting tar %s", archive.name)
        shutil.unpack_archive(str(archive), str(target))
    else:
        logger.info("Not an archive, leaving in place: %s", archive.name)


# ──────────────────────────────────────────────────────────────────────────
# MVTec AD (surface-defect anomaly detection)
# ──────────────────────────────────────────────────────────────────────────
# The 15 official MVTec AD categories (10 objects + 5 textures). Used to verify
# that an extracted dataset matches the layout anomalib's MVTecAD datamodule
# expects: <root>/<category>/{train,test,ground_truth}/...
MVTEC_CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
)


def _download_mvtec(spec: DatasetSpec) -> Path:
    """Download + structure the MVTec AD dataset for use with anomalib.

    MVTec AD is research-license gated, so it can't be pulled anonymously like a
    Roboflow dataset. Supply it via the registry entry one of two ways::

        params:
          url: "https://<your-mirror>/mvtec_anomaly_detection.tar.xz"  # optional
          categories: ["bottle", "cable"]   # optional subset to verify; [] = all 15

    Resolution order:
      1. Target already holds the structured dataset -> nothing to download.
      2. An archive sits in the target dir (you downloaded it manually) -> extract.
      3. A real ``url`` is given -> stream + extract it.
      4. Otherwise raise with instructions (the common license-gated case).

    Result is the canonical MVTec layout anomalib consumes::

        data/raw/anomaly/mvtec_ad/<category>/{train,test,ground_truth}/...
    """
    params = spec.params or {}
    target = spec.target_dir()          # data/raw/anomaly/mvtec_ad
    target.mkdir(parents=True, exist_ok=True)

    if _mvtec_categories_present(target):
        logger.info("MVTec AD already structured at %s — skipping download.", target)
        return _verify_mvtec(target, params.get("categories"))

    archive = _find_local_archive(target)
    url = params.get("url", "") or ""
    if archive is not None:
        logger.info("Using locally provided MVTec archive: %s", archive)
    elif url and not url.startswith("REPLACE_ME"):
        archive = target / Path(url.split("?")[0]).name
        if not archive.exists():
            _stream_download(url, archive)
    else:
        raise RuntimeError(
            "MVTec AD is license-gated and no source was found. Either:\n"
            "  - register + download 'mvtec_anomaly_detection.tar.xz' from "
            "https://www.mvtec.com/company/research/datasets/mvtec-ad and drop it "
            f"into {target}/ , then re-run; or\n"
            "  - host it on a URL you control and set params.url in datasets.yaml."
        )

    _extract_archive(archive, target)
    _flatten_mvtec(target)
    return _verify_mvtec(target, params.get("categories"))


def _mvtec_categories_present(root: Path) -> bool:
    """True if ``root`` already holds the structured dataset (a category w/ train/good)."""
    if not root.exists():
        return False
    return any((sub / "train" / "good").is_dir() for sub in root.iterdir() if sub.is_dir())


def _find_local_archive(root: Path) -> Path | None:
    """Return a manually-downloaded MVTec archive in ``root``, if one is present."""
    for pattern in ("*.tar.xz", "*.tar.gz", "*.tgz", "*.tar", "*.zip"):
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _flatten_mvtec(root: Path) -> None:
    """Hoist category folders up if the archive extracted into a wrapper directory.

    Some MVTec mirrors pack everything under a single top-level folder (e.g.
    ``mvtec_anomaly_detection/``). anomalib expects the categories directly under
    the dataset root, so when we detect that wrapper we move its contents up one
    level and remove it.
    """
    if _mvtec_categories_present(root):
        return  # already flat — nothing to do

    for wrapper in [d for d in root.iterdir() if d.is_dir()]:
        children = [c for c in wrapper.iterdir() if c.is_dir()]
        if any((c / "train" / "good").is_dir() for c in children):
            logger.info("Flattening MVTec wrapper folder: %s/", wrapper.name)
            for child in wrapper.iterdir():
                shutil.move(str(child), str(root / child.name))
            wrapper.rmdir()
            return


def _verify_mvtec(root: Path, categories: list[str] | None = None) -> Path:
    """Log present categories, warn about expected-but-missing ones, return ``root``."""
    found = sorted(
        sub.name for sub in root.iterdir()
        if sub.is_dir() and (sub / "train" / "good").is_dir()
    )
    logger.info("MVTec AD categories present (%d): %s", len(found), ", ".join(found) or "(none)")

    # If the registry names a subset, verify exactly that; otherwise expect all 15.
    expected = list(categories) if categories else list(MVTEC_CATEGORIES)
    missing = [c for c in expected if c not in found]
    if missing:
        logger.warning("MVTec categories expected but missing: %s", ", ".join(missing))

    if not found:
        raise RuntimeError(
            f"No MVTec categories found under {root} after extraction. "
            "Is the archive a valid MVTec AD export?"
        )
    return root


# Registry of source -> handler. Add a new source by writing a function above
# and registering it here; the dispatch loop below needs no changes.
_DOWNLOADERS = {
    "roboflow": _download_from_roboflow,
    "url": _download_from_url,
    "mvtec": _download_mvtec,
}


# ──────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────
def download_dataset(spec: DatasetSpec) -> Path:
    """Download a single dataset, dispatching on its source."""
    handler = _DOWNLOADERS.get(spec.source)
    if handler is None:
        raise ValueError(
            f"No downloader registered for source '{spec.source}' "
            f"(dataset '{spec.name}'). Known sources: {sorted(_DOWNLOADERS)}."
        )
    logger.info("=== %s [%s/%s] ===", spec.name, spec.modality, spec.source)
    return handler(spec)


def download_all(specs: list[DatasetSpec]) -> dict[str, Path]:
    """Download every spec; collect results and keep going past failures."""
    results: dict[str, Path] = {}
    for spec in specs:
        try:
            results[spec.name] = download_dataset(spec)
        except Exception as exc:  # noqa: BLE001 — report and continue the batch
            logger.error("Failed to download '%s': %s", spec.name, exc)
    return results


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download & organize open-source defect-detection datasets.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="download every enabled dataset")
    group.add_argument("--name", type=str, help="download a single dataset by name")
    group.add_argument("--list", action="store_true", help="list registry entries and exit")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="enable debug logging"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config.ensure_dirs()
    specs = config.load_dataset_registry()

    if args.list:
        print(f"{len(specs)} enabled dataset(s) in the registry:")
        for spec in specs:
            print(f"  - {spec.name:<20} [{spec.modality}/{spec.source}]")
        return 0

    if args.name:
        match = next((s for s in specs if s.name == args.name), None)
        if match is None:
            logger.error(
                "No enabled dataset named '%s'. Available: %s",
                args.name,
                ", ".join(s.name for s in specs) or "(none)",
            )
            return 1
        download_dataset(match)
        return 0

    # args.all
    results = download_all(specs)
    logger.info("Done. %d/%d dataset(s) downloaded.", len(results), len(specs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
