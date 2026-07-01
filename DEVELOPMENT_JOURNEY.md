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
7. [Object Detection Training (YOLO)](#7-object-detection-training-yolo)
8. [Wrong-Label Detection (OCR)](#8-wrong-label-detection-ocr)
9. [Current Project State](#9-current-project-state)
10. [Roadmap](#10-roadmap)

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

## 7. Object Detection Training (YOLO)

*(Session of 2026-06-23.)*

### The dataset

With a Roboflow API key in `.env`, `dataset_loader.py` pulled the **PCB Defect Detection (Ultra)** dataset from Roboflow Universe (`cnn-pcb-defect-detection/pcb-defect-detection-ultra`, auto-resolved by our `"latest"` logic to v6). It arrives in YOLOv8 format with the split layout ultralytics consumes directly:

- **6 defect classes:** Missing_Hole, MouseBite, Open_Circuit, Short_Circuit, Spur, Spurious_Cooper
- **6,333 train / 1,778 val / 903 test** images, each split carrying a `data.yaml`

### The training journey (three failures before success)

Getting a clean run on the 4GB laptop GPU took three diagnosed failures — each worth recording for the next session:

**1. GPU vanished mid-session (power management).** The first launch failed with `torch.cuda.is_available(): False` and `nvidia-smi` returning a *permissions* error — even though CUDA had worked earlier the same day. Root cause: the laptop dropped to low battery and **NVIDIA Optimus disabled the discrete GPU to save power**. Fix: back on AC power the RTX 3050 reappeared. Lesson — keep the laptop plugged in for any GPU run.

**2. OpenBLAS RAM exhaustion from stacked workers.** The next run died with `OpenBLAS: Memory allocation still failed after 10 retries`. This is *system RAM*, not VRAM. The cause was **leftover DataLoader worker processes**: the earlier failed runs never cleaned up their 8 workers each, so 16+ zombie `python.exe` processes (~500MB apiece) were simultaneously holding image data on the 16GB machine. Fix: killed the stale processes to reclaim RAM, and **added a configurable `--workers` flag** to `train_yolo.py` (it was hardcoded to ultralytics' default of 8). Re-running with fewer workers trained cleanly.

**3. AutoBatch was too conservative.** Asked to use more of the idle GPU, we tried `--batch -1` (ultralytics AutoBatch). It chose only **batch 5** (targeting 60% VRAM, estimating 1.88GB needed). But our own ground truth from the earlier run showed actual steady-state training used just **0.55GB at batch 4** — AutoBatch's profiler over-estimates peak memory on small cards. We overrode it with an explicit, known-safe `--batch 16`.

### A note on GPU utilisation

A recurring question this session: *"why isn't the GPU at 100%?"* Two reasons. (a) `yolov8n` is a **nano** model (3M params, 8 GFLOPs) — it finishes a batch faster than the CPU can prepare the next one, so the GPU idles between batches. (b) The workload is **data-pipeline-bound** — throughput is capped by how fast the DataLoader workers read+augment JPEGs, not by GPU compute. A bigger batch improves per-step efficiency but cannot exceed the data-loading ceiling. The genuine way to saturate the GPU is a **bigger model** (`yolov8s/m`), which is the right call for the real training run (and also improves accuracy).

### Results

Final config: `yolov8n`, 5 epochs, batch 16, imgsz 640, workers 8, device 0. **Trained in ~12.4 minutes**, peak VRAM ~1.5GB.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **all** | **0.966** | **0.945** | **0.978** | **0.672** |
| Missing_Hole | 0.993 | 0.985 | 0.994 | 0.771 |
| MouseBite | 0.970 | 0.927 | 0.977 | 0.605 |
| Open_Circuit | 0.947 | 0.947 | 0.977 | 0.618 |
| Short_Circuit | 0.979 | 0.960 | 0.984 | 0.686 |
| Spur | 0.961 | 0.925 | 0.966 | 0.702 |
| Spurious_Cooper | 0.947 | 0.923 | 0.971 | 0.650 |

Inference speed: **4.3 ms/image**.

**Honest caveat:** mAP50 of 0.978 after only 5 epochs indicates a relatively clean dataset (train/val likely share visual style). The stricter **mAP50-95 (0.672)** is the number a longer run would mainly improve, by tightening box localisation. This is a real, working detector — but for any production claim it should be retrained for more epochs (ideally `yolov8s` on Colab) and evaluated on held-out, visually distinct data.

### Artifacts

- **Trained weights:** `models/yolo/pcb_defects_yolo/weights/best.pt` (6.2MB, force-added to git; the rest of the run dir — `last.pt`, plots, train/val mosaics — is gitignored for consistency with `models/anomaly/`).
- **Demo:** `demo_yolo.py` runs `best.pt` on a PCB image, draws the predicted boxes via ultralytics' `result.plot()`, and saves an annotated PNG. Verified on a `missing_hole` test image — a clean "Missing_Hole 0.82" box landed exactly on the absent hole.

---

## 8. Wrong-Label Detection (OCR)

The fourth and final defect class — wrong / mislabelled products — needs **no training**. EasyOCR ships pre-trained detection + recognition models, so the task is purely *integration*: read the label text, compare it to what it should say.

### Integration into `DefectInspector`

OCR became the third detector behind the unified `inspect()` API:

- **`enable_ocr=True`** (constructor flag) loads an `easyocr.Reader(["en"], gpu=…)` once. The `gpu` flag is derived from the resolved device (True for any `cuda*`). The first load downloads ~64MB of models to the EasyOCR cache.
- **`inspect(image, expected_text=…)`** gained an optional `expected_text` argument. EasyOCR reads all text via `readtext(np.array(image))`; reads below `min_conf=0.3` are dropped.
- The result dict gained an **`ocr`** block:
  ```python
  "ocr": {
    "text_found": ["ELEC-1", "E36", "E49"],
    "full_text":  "ELEC-1 E36 E49",
    "items":      [{"text": ..., "confidence": ...}, ...],
    "expected":   "ELEC-1" | None,
    "label_ok":   True | False | None,   # None when no expected_text given
  }
  ```

### The comparison logic

OCR output is noisy (case, spacing, stray punctuation), so the match is done on a normalised alphanumeric core: `_normalize_text()` lower-cases and strips everything but letters/digits (`"ELEC-1 "` → `"elec1"`). `_compare_label()` returns True if the normalised expected string is a substring of the normalised OCR text — tolerant of extra text printed around the label. `label_ok=False` means **wrong label detected**.

Like the other detectors, OCR runs in its own try/except — a failure lands in `errors` and never aborts the anomaly/detection paths. `inspect_image.py` (the surface-defect visual demo) passes `enable_ocr=False` so it doesn't pay the OCR load cost it doesn't use.

### Validation

Validated on a **real PCB photo** (`samples/OCR_test-E3330BM.jpg`) by verifying the board's printed model marking:

- **Correct expected** (`"ELEC-1"`): EasyOCR reads `ELEC-1` (with other board text); `label_ok = True`. ✅
- **Wrong expected** (`"ELEC-2"`): `label_ok = False` — wrong label correctly flagged. ✅

**Honest finding:** the board's *etched batch code* `E3330BM` is **misread** (e.g. as `EjJJ0BM`) because the stylized, low-contrast silkscreen font confuses characters (3↔J) — crop/contrast/upscale preprocessing did not fix it. So the demo verifies the clean printed marking `ELEC-1`, not the etched batch code. This is a real, well-known OCR limitation: reliable on printed labels, weaker on small etched markings.

---

## 9. Current Project State

### What Is Complete

| Component | Status | Notes |
|---|---|---|
| `src/config.py` | ✅ Complete | All paths, registry loading, `DatasetSpec` dataclass |
| `src/dataset_loader.py` | ✅ Complete | Roboflow, URL, and MVTec AD handlers; CLI |
| `configs/datasets.yaml` | ✅ Complete | PCB defects (enabled), MVTec AD (enabled), bottle (disabled) |
| MVTec AD extraction | ✅ Complete | `metal_nut` category fully extracted and verified |
| `src/train_anomaly.py` | ✅ Complete | PatchCore/ResNet18 on MVTec AD; CLI; export |
| PatchCore training | ✅ Complete | Image AUROC 0.9946 on metal_nut |
| `src/inference_pipeline.py` | ✅ Complete | `DefectInspector` unifies YOLO + PatchCore + OCR; validated |
| `src/train_yolo.py` | ✅ Complete | Configurable `--workers`; trained on PCB defects |
| YOLO training | ✅ Complete | yolov8n, mAP50 0.978 across 6 PCB defect classes |
| OCR / wrong-label path | ✅ Complete | EasyOCR wired into `DefectInspector`; logic validated |
| Demo tools | ✅ Complete | `inspect_image.py` (anomaly panel) + `demo_yolo.py` (PCB boxes) |
| Gradio demo app | ✅ Complete | `app.py`; mode selector across all 4 defect types; one-click examples |
| Damaged-packaging YOLO detector | ✅ Complete | yolov8n on `packaging_damage`; mAP50 0.810; wired into `DefectInspector` as `yolo_variant="packaging"` |

**All four originally-scoped defect classes now have a working detector** (surface / missing-components / wrong-labels / damaged-packaging). Phase 2 (§11) is about strengthening what's here, not filling gaps.

### What Is Not Yet Built

| Component | Priority | Notes |
|---|---|---|
| Longer / bigger YOLO retrain | Medium | Notebook ready (`notebooks/colab_training.ipynb`); **awaiting user to run it** on Colab |
| More MVTec categories | Low | Only `metal_nut` trained; notebook ready to consume more; **blocked** — license-gated, needs another archive from the user |
| OCR on etched/varied labels | Low | Works on printed markings (ELEC-1); etched silkscreen (E3330BM) misreads |
| Batch/report mode + public deployment | Medium | Phase 2 final step (§11) — folder-of-images CSV/PDF report + Hugging Face Spaces |

### Key File Locations

- **Trained PatchCore model:** `models/anomaly/mvtec_ad_metal_nut_patchcore/weights/torch/model.pt`
- **Trained PCB YOLO model:** `models/yolo/pcb_defects_yolo/weights/best.pt`
- **Trained packaging YOLO model:** `models/yolo/packaging_damage_yolo/weights/best.pt`
- **Base YOLO weights:** `yolov8n.pt` (project root)
- **MVTec AD data:** `data/raw/anomaly/mvtec_ad/metal_nut/`
- **PCB data:** `data/raw/yolo/pcb_defects/`
- **Packaging data:** `data/raw/yolo/packaging_damage/`
- **Dataset registry:** `configs/datasets.yaml`
- **Demo tools:** `inspect_image.py` (anomaly panel), `demo_yolo.py` (PCB boxes), `app.py` (full Gradio UI, all 4 modes)
- **Colab retraining notebook:** `notebooks/colab_training.ipynb` (YOLO scale-up + PatchCore scale-up/more categories — ready to run, not yet executed)

---

## 10. Roadmap

### Next Session

1. **Gradio / Streamlit demo** — wrap `DefectInspector` in an image-upload UI that shows all four results (detections, anomaly heatmap, OCR/label check) in one view. The final deliverable; deployable to Hugging Face Spaces.
2. **Colab notebook** — clone the repo, install deps, and run the full train + evaluate pipeline on a real GPU, free of the 4GB / Optimus constraints hit this session.
3. **Stronger YOLO** — retrain with `yolov8s` and more epochs to lift mAP50-95 and generalisation (this is also the path to genuinely saturating the GPU).

### Medium Term

4. **OCR robustness** — improve reading of low-contrast / etched markings (e.g. preprocessing or higher-res capture); currently reliable on clean printed labels like `ELEC-1`.
5. **More MVTec categories** — the loader + trainer already support all 15; train more once the archives are provided.

### Scaling Considerations

- **PatchCore with `wide_resnet50_2`** on Colab (16GB VRAM): change `--backbone wide_resnet50_2 --image-size 512 --batch 32`. Expected Image AUROC improvement of ~0.5–1% on metal_nut.
- **Multi-category training:** The dataset loader and trainer both support all 15 MVTec AD categories. Training all 15 in a loop is a one-liner once the full archive is available.
- **Custom YOLO dataset:** The `packaging_damage` entry in `datasets.yaml` covers packaging defect detection (see Phase 2 below).

---

## 11. Phase 2 — Post-Presentation Roadmap

*(Started 2026-07-02, after the first successful team presentation.)* The presentation went well; rather than stop, the plan is to go back over the same four ideas raised at the end of Phase 1 and execute them **one at a time, in this order**, each documented here as it lands. Nothing in the sections above is being rewritten — this section (and the ones that follow it) is where ongoing work gets logged, per the standing protocol at the top of this file.

### The Phase 2 plan, in order

1. **Damaged-packaging YOLO detector** — the 4th and last defect class this project set out to cover. Closes the gap left in §9 ("What Is Not Yet Built").
2. **Retrain the PCB YOLO detector on Colab** — bigger backbone (`yolov8s`) and more epochs, to lift mAP50-95 (currently 0.672) beyond what the local 4GB/Optimus-limited GPU could reasonably do.
3. **Expand PatchCore to more MVTec categories** — train 2-3 more categories beyond `metal_nut` (e.g. `bottle`, `cable`) to demonstrate the anomaly-detection approach generalises, ideally with the larger `wide_resnet50_2` backbone on Colab.
4. **Product-ify the demo** — batch/report mode (point at a folder, get a CSV/PDF QC report) and public deployment (e.g. Hugging Face Spaces) so the app isn't tied to one laptop. Deliberately last: it's the polish layer on top of a pipeline that, by this point, will cover all four defect classes with stronger numbers.

### 11.1 Damaged-Packaging Detector — Dataset Selection

Searched Roboflow Universe for a real, downloadable damaged-packaging dataset (the earlier `bottle_inspection` entry in `datasets.yaml` was only ever a placeholder with `REPLACE_ME` slugs — never a real dataset). Evaluated three real candidates:

| Candidate | Workspace/project | Classes | Size | Verdict |
|---|---|---|---|---|
| `object-detection-5pf5v/packaging-defect-detection-wbcpk` | 2 classes: `Box`, `date` | — | Not damage-related (box + date-label detection) — rejected |
| `packaging-defect-detection/package-defect-detection-loin8` | 2 classes: `defected`, `non_defected` | — (size unconfirmed) | Clean framing but couldn't confirm size/splits |
| **`iot-project/damaged-package-detection`** ✅ chosen | 5 classes (damaged / damaged food packaging box / food item boxes / packaging boxes / packaging boxes that are damaged) | **~1,000 images** (794 train / 103 valid / 103 test) | Chosen — real, verified stats, free via Roboflow API |

Picked `iot-project/damaged-package-detection` — it's the one candidate with concrete, verified image counts and split sizes rather than an unconfirmed page. The class list looks messier than the PCB dataset's on paper, but Roboflow's own preprocessing note ("Modify Classes: 4 remapped, 0 dropped") suggests the effective label set is smaller in practice; the real breakdown is captured from the downloaded `data.yaml`, not hand-curated here.

Wired into `configs/datasets.yaml` as a new `packaging_damage` entry (replacing the old placeholder), same pattern as `pcb_defects`: `source: roboflow`, `version: "latest"`.

**A real download failure worth recording:** the first candidate actually attempted, `iot-project/damaged-package-detection` (the one with the cleanest-looking stats), turned out to be a Roboflow **classification** project, not object detection — `dataset_loader.py` connected fine and resolved `version: "latest" -> v4`, but the export call failed with `"yolov8 is an invalid format for project type classification. Please use one of: folder, clip."` Roboflow's own project-type metadata isn't visible from search snippets, so this was only caught by actually attempting the download. Switched to `packaging-defect-detection/package-defect-detection-loin8` (confirmed real object-detection project, CC BY 4.0), which downloaded cleanly.

**Downloaded dataset:** 2,852 images (2,496 train / 240 valid / 116 test), 2 classes (`defected`, `non_defected`). Spot-checked label files across several images — most have multiple, differently-positioned/sized boxes per image (not just one full-frame box), confirming this is genuine localized object detection, not a classification dataset dressed up as one. Filename analysis showed 207 unique source photos in the train split, each augmented ~12× by Roboflow (standard flip/rotate/brightness augmentation) — a normal and expected pattern, not a red flag. Visual spot-check of the raw images: consistently dark/grainy, low-light photography (e.g. one frame shows a visible crack line in a jar body near its cap) — genuine but not high-production-value imagery.

### 11.2 Damaged-Packaging Detector — Training & Results

Trained with the same proven settings as the PCB detector (`yolov8n`, batch 16, imgsz 640, workers 8, 5 epochs, device 0):

```bash
python -m src.train_yolo --dataset packaging_damage --epochs 5 --batch 16 --imgsz 640 --workers 8 --device 0 --name packaging_damage_yolo
```

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **all** | **0.646** | **0.871** | **0.810** | **0.789** |
| defected | 0.775 | 0.750 | 0.865 | 0.843 |
| non_defected | 0.518 | 0.991 | 0.755 | 0.736 |

**Honest read:** overall mAP50 0.81 is solid for a 5-epoch smoke test, but precision on `non_defected` (0.518) is notably weaker than the PCB detector's numbers — the model over-predicts `non_defected` (very high recall, 0.991, but more false positives), which tracks with the dataset's dark/grainy imagery and modest source-photo diversity (207 unique photos). `defected` — the class that actually matters for catching real damage — has better precision (0.775) and a strong mAP50 (0.865). Same caveat as the PCB run applies: this is a smoke test, not a final production model; the Colab retrain (task 2 of the Phase 2 plan) is where both detectors get a real epoch count and a bigger backbone.

Weights: `models/yolo/packaging_damage_yolo/weights/best.pt`.

### 11.3 Wiring a Second YOLO Model into `DefectInspector`

`DefectInspector` was built around a single YOLO model. Adding a second, domain-specific detector (packaging) without breaking the existing PCB path needed a small, backward-compatible extension rather than a rewrite:

- **`__init__`** gained an optional `packaging_yolo_weights` param. When given, a second `ultralytics.YOLO` instance is loaded into `self._packaging_yolo` alongside the existing `self._yolo` (PCB). When omitted, behaviour is unchanged.
- **`inspect()`** gained a `yolo_variant: str = "pcb"` param. `"pcb"` (the default) runs the primary model exactly as before; `"packaging"` runs the packaging model if one was loaded, silently falling back to the primary model otherwise (so old call sites never break).
- **`_run_yolo()`** now accepts an optional `model=` argument (defaulting to `self._yolo`), so the same detection-parsing logic serves both models without duplication.

This is deliberately minimal — no plugin registry, no dict-of-models abstraction — because there are exactly two YOLO use cases today and the parameter approach reads clearly at both call sites. If a third YOLO domain shows up later, that's the point to generalise to a named-model dict.

**In `app.py`:** a new `MODE_PACKAGING` mode was added alongside the existing three. The existing `_pcb_view()` renderer (boxes + class/confidence table) is fully generic over `result["detections"]`, so it's reused as-is for packaging — no new view function needed. The mode dispatch in the `inspect()` callback picks `yolo_variant="packaging"` when `mode == MODE_PACKAGING`, otherwise `"pcb"`. An example image was chosen by scanning the whole test set for the highest-confidence `defected` prediction (found at conf 0.89) rather than picking arbitrarily — same principle as the PCB "Missing_Hole" example from Phase 1.

Verified end-to-end through the running app (not just unit-level): packaging image → 1 defect detected; PCB image → unchanged regression check still passes (confirming the two-model routing doesn't cross-contaminate).

### 11.4 Colab Retraining Notebook

With all four detectors trained locally at smoke-test scale, the next Phase 2 item is retraining with bigger models/longer schedules on Colab's free GPU. This is split cleanly by what's actually executable from here versus what needs the user's own hands:

- **Preparing the notebook** — fully within scope; done in this session.
- **Actually running it** — needs the user's own Google account and an interactive Colab session in their browser. This is *not* something that can be driven headlessly/autonomously (no credentials, and a multi-epoch training run isn't a sensible thing to babysit via browser automation). So this task's honest completion state is: **notebook ready, execution pending on the user.**

Built `notebooks/colab_training.ipynb` as a single notebook covering both remaining Colab-shaped Phase 2 items at once (rather than two notebooks that would each repeat the same clone/install/mount overhead):

- **Setup** — clone the repo from GitHub, `pip install -r requirements.txt` (Colab's own CUDA-enabled torch already satisfies the `torch>=2.2.0` pin, so it's left untouched — no repeat of the CPU/CUDA wheel confusion from Phase 1).
- **Secrets** — a placeholder cell for the Roboflow API key, with an explicit warning not to commit a real key after filling it in.
- **Part 1 (YOLO)** — retrains both `pcb_defects` and `packaging_damage` with `yolov8s` (vs. `yolov8n` locally) for 100 epochs (vs. 5). Deliberately uses the **same run names** as the local training (`pcb_defects_yolo`, `packaging_damage_yolo`) so bringing weights home is a plain file overwrite — no path changes needed in `app.py` or anywhere else.
- **Part 2 (PatchCore)** — mounts Google Drive, copies in an MVTec archive the user has to supply themselves (still license-gated — Colab doesn't remove that constraint), then retrains `metal_nut` with `wide_resnet50_2` at 512px (vs. `resnet18` at 256px locally). A 2b sub-section loops over an editable list of additional categories for whenever the user provides more archives — this is the mechanism that will eventually satisfy the "more MVTec categories" roadmap item, but it stays inert (`EXTRA_CATEGORIES = []`) until archives exist.
- **Download** — zips `models/` and downloads it, with explicit instructions on which files map to which local paths.

**What's blocked and why:** the "expand PatchCore to more categories" task cannot progress further right now — MVTec AD requires registering on MVTec's site and downloading category archives under their research license, which is not something that can be done anonymously or without the user's credentials. The notebook is ready to consume more categories the moment archives exist; nothing else needs to change.

### 11.5 Bug Fix: `_download_mvtec` Couldn't Add Categories Incrementally

Caught while preparing to actually add more categories: `_download_mvtec()`'s very first check was `_mvtec_categories_present(target)` — true the moment *any* category (e.g. the existing `metal_nut`) is extracted — which returned immediately, **before ever looking for new archives**. Dropping `bottle.tar.xz` next to an already-extracted `metal_nut/` and re-running the loader would have silently done nothing. Compounding this, the archive finder (`_find_local_archive`, singular) only ever returned the *first* matching archive it found, ignoring any others.

Fixed both issues:
- `_find_local_archives` (plural) now returns every local archive, not just one.
- `_download_mvtec` extracts each archive whose category isn't already present, regardless of what else is already extracted. A new `_infer_archive_category()` helper matches an archive's filename against the known 15 MVTec category names (e.g. `bottle.tar.xz` -> `"bottle"`) to decide whether to skip it.
- Every processed archive (extracted *or* skipped) gets renamed with an `.extracted` suffix so it's never re-scanned — this also sidesteps a real Windows `PermissionError` hit while testing the fix: re-extracting an archive into an *already-populated* folder can fail on files Windows has locked/marked read-only from the original extraction. Skipping already-done categories entirely (rather than re-extracting them) avoids ever touching those files again.

Verified: re-ran the loader against the existing `metal_nut.tar.xz` (never renamed from the original Phase 1 session) — it correctly skipped extraction, renamed the archive to `metal_nut.tar.xz.extracted`, left the 220 `train/good` images untouched, and a second re-run correctly fell through to the "already structured" no-op path.
