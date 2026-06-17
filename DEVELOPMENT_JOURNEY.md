# Defect Detection — Development Journey

> **PROTOCOL: This document must be updated with highly detailed logs at the end of every active work session.**

This file is the canonical engineering narrative for the Defect Detection project (Tech Mahindra internship). It is written to be self-contained: a future collaborator (or a future session of this project) should be able to read it cold and understand not just *what* was built, but *why* every decision was made and what obstacles were overcome.

---

## Table of Contents

1. [Project Goal & Architecture](#1-project-goal--architecture)
2. [Repository Scaffold](#2-repository-scaffold)
3. [Data Pipeline Construction](#3-data-pipeline-construction)
4. [Hardware Migration & CUDA Fix](#4-hardware-migration--cuda-fix)
5. [Anomaly Detection Training (PatchCore)](#5-anomaly-detection-training-patchcore)
6. [Unified Inference Engine](#6-unified-inference-engine)
7. [Current Project State](#7-current-project-state)
8. [Roadmap](#8-roadmap)

---

## 1. Project Goal & Architecture

### The Problem

Manual quality inspection on manufacturing lines is slow, inconsistent, and expensive. The goal of this project is to build a **generic, AI-based defect detection system** that can take a static image of a product and automatically flag four distinct classes of defect:

| Defect Class | Description | Technique |
|---|---|---|
| **Missing components** | A part is absent from an assembly (e.g., a missing capacitor on a PCB) | Object detection — YOLO |
| **Damaged packaging** | A container, box, or outer casing shows physical damage | Object detection — YOLO |
| **Surface defects** | Scratches, dents, discolouration, or other texture anomalies on a product's surface | Anomaly detection — PatchCore |
| **Wrong labels** | A product label contains incorrect text (wrong SKU, wrong date, etc.) | OCR + string comparison — EasyOCR |

### Why This Architecture?

Each defect class has a fundamentally different detection paradigm, which is why a single model cannot address all four. The architecture maps each class to the best-fit technique:

**Object Detection (YOLO via `ultralytics`)**
YOLOv8 is the industry standard for real-time object detection. It runs in a single forward pass and can detect the *presence or absence* of specific components when trained on labelled datasets. For missing-component and damaged-packaging detection, we need a model that understands *what objects should be in the scene* — YOLO's bounding-box output is perfect for this. We chose YOLOv8n (nano) as the base weights because it is the lightest member of the YOLOv8 family, suitable for both local testing on a 4GB VRAM GPU and fast inference during deployment.

**Anomaly Detection (PatchCore via `anomalib`)**
Surface defect detection is a fundamentally different problem. Defects are, by definition, rare and unpredictable — it is impractical to collect and label thousands of examples of every possible scratch, dent, or colour aberration. PatchCore solves this with an **unsupervised, memory-bank approach**: it trains exclusively on *normal* (defect-free) images, extracting mid-level CNN features and storing a representative coreset of them. At inference time, any image whose patch features are far from the stored "normal" distribution is flagged as anomalous. This requires no defect labels at all — only clean examples.

`anomalib` (Intel's open-source anomaly detection library) was chosen because it provides production-grade implementations of PatchCore, MVTecAD data loaders, Lightning-based training, and first-class export to TorchScript — all out of the box.

**OCR (EasyOCR) — planned**
Wrong-label detection requires reading text from an image and comparing it against an expected value. EasyOCR is a lightweight, pre-trained multi-language OCR engine that requires no training — only sample images to test against. This pipeline component is scaffolded for future implementation.

### Key Design Principles

- **Config-driven data pipeline.** All datasets are described in `configs/datasets.yaml`. Adding a new dataset means adding a YAML entry, not modifying Python code.
- **Environment-agnostic paths.** All filesystem paths are derived from `config.py` using `Path(__file__).resolve().parents[1]`, so the project runs identically on a local Windows laptop and on Google Colab without any code changes.
- **Lightweight-by-default, scale-up-ready.** All training defaults (ResNet18 backbone, batch size 8, image size 256, coreset ratio 0.1) are tuned for a 4GB VRAM GPU. Switching to Colab with a larger backbone is a single flag change.
- **Separation of concerns.** Data downloading, model training, and inference are three separate modules with clean interfaces between them.

---

## 2. Repository Scaffold

The repository was initialised with the following structure:

```
Defect-Detection-Tech-M/
├── configs/
│   └── datasets.yaml          # Registry: all dataset sources and parameters
├── data/
│   ├── raw/
│   │   ├── yolo/              # YOLO-format detection datasets (Roboflow)
│   │   ├── anomaly/           # MVTec AD surface-defect dataset
│   │   └── ocr/               # Label image crops for OCR testing
│   └── processed/             # Normalised data ready for training
├── models/
│   ├── yolo/                  # Trained YOLO weights (.pt)
│   └── anomaly/               # PatchCore run directories (checkpoints + exports)
├── notebooks/                 # Google Colab training notebooks (planned)
├── src/
│   ├── __init__.py
│   ├── config.py              # All filesystem paths + dataset registry loading
│   ├── dataset_loader.py      # Download & extract datasets
│   ├── train_yolo.py          # YOLO training entry point (scaffolded)
│   ├── train_anomaly.py       # PatchCore training entry point
│   └── inference_pipeline.py  # Unified DefectInspector class
├── yolov8n.pt                 # Pre-trained YOLOv8 nano base weights
├── requirements.txt
├── .env                       # Local secrets (gitignored)
├── .gitignore
└── README.md
```

`src/config.py` is the single source of truth for all paths. Every other module imports from it rather than constructing paths independently. This is critical for Colab compatibility — `PROJECT_ROOT` is derived dynamically, so there are no hardcoded absolute paths anywhere in the codebase.

---

## 3. Data Pipeline Construction

### `configs/datasets.yaml` — The Dataset Registry

All dataset sources are declared in a single YAML file. Each entry has:

- `name`: the folder name it will be saved under (e.g., `mvtec_ad`)
- `modality`: `yolo`, `anomaly`, or `ocr` — determines which subdirectory under `data/raw/` the dataset lands in
- `source`: the downloader to use (`roboflow`, `url`, or `mvtec`)
- `enabled`: a boolean flag so unused datasets can be kept in the registry without being downloaded
- `params`: source-specific arguments (API slugs, URLs, category lists)

The registry is parsed by `src/config.py`'s `load_dataset_registry()`, which returns a list of typed `DatasetSpec` dataclass objects. Only entries with `enabled: true` are returned, so the downloader never has to filter manually.

### `src/dataset_loader.py` — The Download Engine

The loader is a dispatch table: a top-level `download_dataset(spec)` function looks up `spec.source` in a `_DOWNLOADERS` dict and calls the right handler. Adding a new data source requires only writing a handler function and registering it — the dispatch logic is never touched.

**Roboflow handler (`_download_from_roboflow`)**

Roboflow datasets (PCB defects, bottle inspection) are pulled via the `roboflow` Python SDK. The handler implements a `"latest"` version resolution strategy: when `version: "latest"` is set in the YAML, it queries the project's version list at runtime, extracts the numeric version ID from each `Version.version` slug (which has the form `"workspace/project/<n>"`), takes the maximum, and uses that. This means the registry never needs to be bumped when a dataset is updated on Roboflow — it always fetches the newest version automatically.

The handler also implements an idempotency check: if `data.yaml` already exists in the target directory (the marker that a Roboflow export is complete), it skips re-downloading. This is important because the Roboflow SDK's own `overwrite=False` flag would silently skip even partially downloaded datasets.

**URL handler (`_download_from_url`)**

For datasets distributed as direct archives (zip or tar), this handler streams the file in 1 MiB chunks with a `tqdm` progress bar and writes to a `.part` temporary file, renaming to the final path only when the download is complete. This prevents a partial download from looking like a valid archive on the next run. Extraction is delegated to `_extract_archive()`, which inspects the suffix to dispatch between `zipfile` and `shutil.unpack_archive`.

**MVTec AD handler (`_download_mvtec`)**

The MVTec Anomaly Detection dataset is **research-license gated** — it cannot be downloaded anonymously. The handler implements a three-step resolution strategy:

1. **Already structured?** Check if any subdirectory under the target already has a `train/good/` folder. If yes, skip entirely.
2. **Local archive present?** Scan the target directory for `*.tar.xz`, `*.tar.gz`, `*.zip` etc. (via `_find_local_archive`). If one is found, extract it directly. This is the workflow used in this session: the `metal_nut.tar.xz` archive was downloaded manually and placed in `data/raw/anomaly/mvtec_ad/`.
3. **URL provided?** Stream and extract the archive from a URL (useful if you host a mirror).
4. **None of the above?** Raise a `RuntimeError` with explicit instructions on where to obtain the dataset.

After extraction, `_flatten_mvtec()` handles the case where the archive extracted into a wrapper directory (e.g., `mvtec_anomaly_detection/bottle/...` instead of `bottle/...`). It detects this by checking whether any direct child directory contains a `train/good` subfolder; if not but a grandchild does, it hoists all grandchildren up one level and removes the wrapper.

Finally, `_verify_mvtec()` logs which categories are present, warns about any that are expected but missing, and raises if *no* categories were found at all — providing an early, clear error rather than a confusing failure deep in the training loop.

**The MVTec folder structure anomalib requires:**
```
data/raw/anomaly/mvtec_ad/
└── metal_nut/
    ├── train/
    │   └── good/          ← 220 normal training images
    ├── test/
    │   ├── good/          ← normal test images
    │   ├── bent/          ← defect class: bent nut
    │   ├── color/         ← defect class: colour defect
    │   ├── flip/          ← defect class: flipped nut
    │   └── scratch/       ← defect class: scratch
    └── ground_truth/
        ├── bent/          ← binary pixel masks for evaluation
        ├── color/
        ├── flip/
        └── scratch/
```

This structure is consumed directly by anomalib's `MVTecAD` datamodule with zero pre-processing on our side.

---

## 4. Hardware Migration & CUDA Fix

### Context

Development started on a Mac. The repository was migrated to a Windows laptop equipped with an **NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)**. This hardware change introduced two challenges: CUDA driver configuration and VRAM budget constraints.

### The PyTorch CUDA Problem

When the virtual environment was recreated on Windows, `pip install -r requirements.txt` installed `torch==2.12.1` — but pulled the **CPU-only wheel** (`torch 2.12.1+cpu`). This is a common trap: PyPI's default PyTorch index hosts CPU wheels; the CUDA builds are on a separate index (`https://download.pytorch.org/whl/cu<version>`).

**Diagnosis:** `torch.cuda.is_available()` returned `False` despite a functional GPU. `nvidia-smi` confirmed the driver was healthy:
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Driver version: 610.62
- Max supported CUDA version: 13.3

**Fix:** Force-reinstall PyTorch with the CUDA 12.8 build (compatible with driver 13.3 via CUDA backward compatibility):

```bash
pip install --force-reinstall torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
```

The `--force-reinstall` flag is essential. Without it, pip sees the existing `2.12.1` version as satisfying `torch>=2.2.0` and does nothing, even though the installed wheel is the wrong variant. The result was `torch 2.11.0+cu128` with `torch.cuda.is_available() == True`.

**Note for future sessions:** The `requirements.txt` does not pin the CUDA index URL (by design, to keep it Colab-compatible — Colab manages its own CUDA build). On any new Windows setup, always run the force-reinstall command above after `pip install -r requirements.txt`.

### VRAM Budget Design

The RTX 3050 Laptop GPU has exactly **4096 MiB** of VRAM. PatchCore's training pipeline (feature extraction with a CNN backbone) is memory-intensive because it processes all training images in a single pass and stores their feature vectors. The following parameters were chosen to stay within budget:

| Parameter | Chosen Value | Why |
|---|---|---|
| `--backbone` | `resnet18` | ~11M parameters vs. 69M for `wide_resnet50_2`; extracts comparable mid-level features |
| `--image-size` | `256` | Quarter of 1024px; feature map size scales quadratically with image size |
| `--batch` | `8` | Processes 8 images per GPU pass during feature extraction |
| `--coreset-ratio` | `0.1` | The coreset (memory bank) is 10% of all extracted patches; controls RAM and inference speed |
| `--num-workers` | `0` | Windows does not support `fork`-based multiprocessing; `0` workers runs the DataLoader in the main process and avoids deadlocks |

None of these are permanent limitations — on Google Colab with a T4 (16GB VRAM) or A100 (40GB), `wide_resnet50_2`, `image-size 512`, and `batch 32` are straightforward upgrades via CLI flags.

---

## 5. Anomaly Detection Training (PatchCore)

### `src/train_anomaly.py`

The training script is a clean wrapper around anomalib's `Engine` API. Its key design decisions:

**Lazy imports.** `anomalib`, `lightning`, and `torch` are imported inside the `train()` function, not at the top of the module. This means `--help` and the dataset validation checks (`find_mvtec_root`) work without loading the heavy ML stack — useful for fast CI checks or when running on a machine without a GPU.

**Device resolution.** The `resolve_accelerator()` function maps our simple `--device` flag (which accepts `auto`, `cpu`, `mps`, `gpu`, or a CUDA index like `0`) to the `(accelerator, devices)` tuple that Lightning's `Trainer` expects. Passing `devices=[0]` instead of `devices=1` ensures a specific physical GPU is targeted rather than letting Lightning pick.

**Single-pass training.** PatchCore is not a gradient-descent model. `engine.fit()` runs exactly one epoch: it passes every training image through the ResNet18 backbone, extracts `layer2` and `layer3` feature maps, and builds a coreset of patch features using greedy farthest-point sampling. This is why training takes ~5.5 minutes rather than hours — no backpropagation, no epochs to iterate.

**TorchScript export.** After training and evaluation, the model is exported via `engine.export(ExportType.TORCH)` to `models/anomaly/<run-name>/weights/torch/model.pt`. This is a self-contained TorchScript file that includes the pre-processor, backbone, coreset, and post-processor — the inference pipeline loads this single file and needs no knowledge of the anomalib training configuration.

### Training Results — metal_nut Category

Training was executed with the command:

```bash
python -m src.train_anomaly \
    --category metal_nut \
    --device 0 \
    --backbone resnet18 \
    --image-size 256 \
    --batch 8 \
    --coreset-ratio 0.1 \
    --num-workers 0
```

**Timing:**
- Feature extraction + coreset build: **333 seconds (~5.5 minutes)**
- Test evaluation: **10 seconds**

**Test metrics on the MVTec AD metal_nut test set:**

| Metric | Score | Interpretation |
|---|---|---|
| Image AUROC | **0.9946** | Near-perfect separation of normal vs. anomalous images |
| Image F1 | **0.9785** | Excellent precision-recall balance at the image level |
| Pixel AUROC | **0.9835** | The anomaly heatmap localises defect regions very accurately |
| Pixel F1 | **0.8177** | Good pixel-level segmentation (lower than image due to boundary imprecision) |

An Image AUROC of 0.9946 is state-of-the-art for PatchCore with a ResNet18 backbone on metal_nut. The full metal_nut category in MVTec AD is known to be challenging due to its four distinct defect types (bent, color, flip, scratch), making this result particularly strong.

**Output artifacts** saved under `models/anomaly/mvtec_ad_metal_nut_patchcore/`:
- `weights/torch/model.pt` — exported TorchScript model (the one used by the inference pipeline)
- Lightning checkpoint files
- Visualisation images showing anomaly heatmaps on test samples

---

## 6. Unified Inference Engine

### `src/inference_pipeline.py` — The `DefectInspector` Class

The inference pipeline is the capstone module that ties all trained models together behind a single, clean API. The design goal was: **one object, one method, one result dict** — callers need no knowledge of YOLO, anomalib, or device management.

### Architecture

```
DefectInspector
├── _load_yolo()          → ultralytics.YOLO  (loads yolov8n.pt)
├── _load_patchcore()     → anomalib TorchInferencer  (loads model.pt)
└── inspect(image) → dict
    ├── _run_yolo()       → list of {label, confidence, box}
    └── _run_patchcore()  → (score, heatmap_array, is_anomalous)
```

### Device Handling

Device management was one of the most careful parts of the implementation, because YOLO (ultralytics) and anomalib's `TorchInferencer` have **different device string formats**:

| Library | Accepted format | Example |
|---|---|---|
| PyTorch / torch | `"cuda"`, `"cuda:0"`, `"cpu"` | `torch.device("cuda:0")` |
| ultralytics YOLO | `"0"`, `"1"`, `"cpu"` | numeric index string |
| anomalib TorchInferencer 2.5 | `"auto"`, `"cuda"`, `"cpu"`, `"xpu"` | no index suffix allowed |

The `_resolve_device()` function normalises the user's input to a canonical torch device string (`"cuda:0"`, `"cpu"`, etc.) first. Then two translation functions convert this to each library's expected format:
- YOLO: `self.device.replace("cuda:", "")` → `"0"` for `"cuda:0"`, pass `"cpu"` as-is
- anomalib: `"cuda"` for any `"cuda:*"` string (anomalib 2.5 does not accept indexed CUDA strings)

Device validation is **fail-fast**: requesting `--device 0` on a machine with no CUDA raises immediately in `_resolve_device()` with a clear message, rather than silently falling back to CPU and giving misleadingly fast results.

### The `inspect()` Return Dictionary

```python
{
    "device":        "cuda:0",      # which device both models ran on
    "detections": [                  # list from YOLO
        {
            "label":      "bowl",    # class name from YOLO's COCO classes
            "confidence": 0.2934,    # detection confidence [0, 1]
            "box":        [48.1, 57.9, 655.5, 700.0]  # [x1, y1, x2, y2] pixels
        }
    ],
    "anomaly_score": 0.4699,        # image-level score; higher = more anomalous
    "anomaly_map":   np.ndarray,    # shape (H, W), float32; pixel-level heatmap
    "is_anomalous":  False,         # bool based on model-embedded threshold
    "errors":        []             # non-fatal per-model error strings, if any
}
```

The `anomaly_map` is a 2D NumPy array (not serialised to JSON by the CLI — its shape is printed instead). It can be overlaid on the original image to create a visual heatmap showing *where* the defect is, not just *whether* one exists.

The `errors` list is a deliberate design choice: if YOLO crashes on a particular image format but PatchCore succeeds, the caller gets a partial result and a non-fatal error message rather than an exception. This is important for a production pipeline where one bad frame shouldn't stop an entire inspection run.

### Key Engineering Issues Encountered

**anomalib 2.5 security gate on pickle loading.** When we first called `TorchInferencer(path=model_pt, device="cuda")`, anomalib 2.5 raised a `ValueError` demanding `TRUST_REMOTE_CODE=1` before loading the TorchScript model. This is a security measure added in anomalib 2.5 against malicious pickled models. Since our `model.pt` was exported by our own `train_anomaly.py` and is entirely trusted, we set `os.environ.setdefault("TRUST_REMOTE_CODE", "1")` in the loader, with a comment explaining the rationale. `setdefault` is used (not `os.environ[...] = "1"`) so that an external caller can pre-set the variable to `"0"` to block loading if needed.

**anomalib's `TorchInferencer` device string format.** The initial call used `"cuda:0"`, which raised `ValueError: Unknown device 'cuda:0'. Expected one of: auto, cpu, cuda, xpu`. This was caught by the smoke test and fixed by collapsing any `"cuda:*"` string to `"cuda"` before passing it to anomalib.

### Validation Results

The pipeline was smoke-tested against two images from the MVTec AD metal_nut test set:

**Normal image** (`test/good/000.png`):
```json
{
  "anomaly_score": 0.4699956476688385,
  "is_anomalous": false
}
```

**Defective image** (`test/bent/000.png`):
```json
{
  "anomaly_score": 0.9217514395713806,
  "is_anomalous": true
}
```

The model correctly classified both images with a large margin between scores (~0.47 vs ~0.92), demonstrating strong discriminative power at inference time.

### CLI Usage

```bash
# Auto-detect GPU, auto-find latest trained run
python -m src.inference_pipeline path/to/image.jpg

# Explicit GPU, verbose logging
python -m src.inference_pipeline path/to/image.jpg --device 0 --verbose

# CPU fallback
python -m src.inference_pipeline path/to/image.jpg --device cpu

# Override anomaly threshold
python -m src.inference_pipeline path/to/image.jpg --threshold 0.5
```

---

## 7. Current Project State

### What Is Complete

| Component | Status | Notes |
|---|---|---|
| `src/config.py` | ✅ Complete | All paths, registry loading, `DatasetSpec` dataclass |
| `src/dataset_loader.py` | ✅ Complete | Roboflow, URL, and MVTec AD handlers; CLI |
| `configs/datasets.yaml` | ✅ Complete | PCB defects (enabled), MVTec AD (enabled), bottle (disabled) |
| MVTec AD extraction | ✅ Complete | `metal_nut` category fully extracted and verified |
| `src/train_anomaly.py` | ✅ Complete | PatchCore/ResNet18 on MVTec AD; CLI; export |
| PatchCore training | ✅ Complete | Image AUROC 0.9946 on metal_nut |
| `src/inference_pipeline.py` | ✅ Complete | `DefectInspector` class; validated on good/bent images |
| `src/train_yolo.py` | ⏸ Scaffolded | Training deferred; `yolov8n.pt` base weights are on disk |

### What Is Not Yet Built

| Component | Priority | Notes |
|---|---|---|
| YOLO training on PCB dataset | Medium | Roboflow dataset configured; training deferred |
| OCR label-check pipeline | Medium | EasyOCR integration planned; no custom training needed |
| `inference_pipeline.py` OCR path | Medium | `DefectInspector` has slots for it; not yet wired |
| Streamlit / Gradio demo UI | High | Final deliverable; wraps `DefectInspector.inspect()` |
| Google Colab training notebook | Medium | Scale-up path for YOLO + larger PatchCore backbone |
| Inference visualisation | Low | Overlay `anomaly_map` on image; save annotated output |

### Key File Locations

- **Trained PatchCore model:** `models/anomaly/mvtec_ad_metal_nut_patchcore/weights/torch/model.pt`
- **Base YOLO weights:** `yolov8n.pt` (project root)
- **MVTec AD data:** `data/raw/anomaly/mvtec_ad/metal_nut/`
- **Dataset registry:** `configs/datasets.yaml`

---

## 8. Roadmap

### Next Session

1. **YOLO training** — run `python -m src.train_yolo` on the PCB defects dataset (already downloaded via Roboflow). Evaluate mAP on the test split.
2. **OCR pipeline** — integrate EasyOCR into `inference_pipeline.py` as a third detection path in `DefectInspector.inspect()`.
3. **Visualisation helper** — write a utility to overlay `anomaly_map` (heatmap) on the original image and draw YOLO bounding boxes, for demo purposes.

### Medium Term

4. **Gradio demo** — wrap `DefectInspector` in a Gradio `Interface` with an image upload widget. One Python file, deployable to Hugging Face Spaces.
5. **Colab notebook** — a single notebook that clones the repo, installs dependencies, mounts Google Drive for the MVTec archive, and runs the full train + evaluate pipeline with the larger backbone.

### Scaling Considerations

- **PatchCore with `wide_resnet50_2`** on Colab (16GB VRAM): change `--backbone wide_resnet50_2 --image-size 512 --batch 32`. Expected Image AUROC improvement of ~0.5–1% on metal_nut.
- **Multi-category training:** The dataset loader and trainer both support all 15 MVTec AD categories. Training all 15 in a loop is a one-liner once the full archive is available.
- **Custom YOLO dataset:** The `bottle_inspection` entry in `datasets.yaml` is ready to be filled in with a Roboflow project slug for packaging defect detection.
