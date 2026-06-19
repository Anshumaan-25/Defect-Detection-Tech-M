# Live Demo Guide — Defect Detection Pipeline

How to run the pipeline live, feed it any image, and show a clean annotated result.

---

## TL;DR (the one command)

```bash
python inspect_image.py <path-to-image>
```

This loads the trained models, runs inference, prints the verdict in the terminal,
and **saves + automatically opens** an annotated result panel.

---

## Before the demo (one-time setup check)

Make sure the trained anomaly model exists and the GPU is live:

```bash
# 1. Confirm CUDA is available (should print: True)
python -c "import torch; print(torch.cuda.is_available())"

# 2. Confirm the trained PatchCore model is on disk
#    -> models/anomaly/mvtec_ad_metal_nut_patchcore/weights/torch/model.pt
```

> **If `torch.cuda.is_available()` is `False`** (e.g. after a fresh `pip install`),
> reinstall the CUDA build:
> ```bash
> pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
> ```
> The demo still works on CPU (`--device cpu`), just a few seconds slower per image.

---

## Running the demo

### Inspect any image

```bash
python inspect_image.py data/raw/anomaly/mvtec_ad/metal_nut/test/bent/000.png
```

Terminal output:
```
==================================================
  Defect Detection -- Single Image Inspection
==================================================
  Image : 000.png
  [1/3] Loading models (YOLO + PatchCore)...
  [2/3] Running inference...
        Objects detected : 0
        Anomaly score    : 0.9218
        VERDICT          : DEFECTIVE
  [3/3] Rendering result panel...
        Saved -> result_000.png
==================================================
```

The result image opens automatically in your default viewer.

### The result panel

A single wide image with four parts:

```
┌─────────────────────────────────────────────────────────────────┐
│  Defect Inspection | <filename>     PatchCore + YOLOv8n          │  ← header
├──────────────────┬──────────────────┬──────────────────────────┤
│  Original        │  Heatmap Overlay │  Pure Anomaly Map         │  ← 3 panels
│  + YOLO boxes    │  (hot = defect)  │  (where the model looks)  │
├──────────────────┴──────────────────┴──────────────────────────┤
│           VERDICT:  DEFECTIVE        Anomaly score: 0.9218        │  ← banner
└─────────────────────────────────────────────────────────────────┘
```

- **Green banner = NORMAL**, **Red banner = DEFECTIVE** (driven by the model's threshold).
- The **heatmap overlay** shows *where* on the part the anomaly is — red = anomalous.
- A **normal part stays cool blue**; a defect lights up red. The colour scale is
  fixed so brightness is comparable between images (a green verdict won't show a
  scary red map).

---

## Useful flags

| Flag | What it does | Example |
|---|---|---|
| `--device` | `auto` / `cpu` / `0` (GPU index) | `--device cpu` |
| `--out` | Where to save the result PNG | `--out demo1.png` |
| `--no-open` | Don't auto-open the image (for scripted runs) | `--no-open` |
| `--threshold` | Override the NORMAL/DEFECTIVE cutoff | `--threshold 0.5` |

---

## Suggested demo script (what to say + do)

1. **Set the stage.** "This pipeline takes any product image and flags surface
   defects using an anomaly-detection model trained only on *good* examples —
   it has never seen a labelled defect."

2. **Show a normal part first.**
   ```bash
   python inspect_image.py data/raw/anomaly/mvtec_ad/metal_nut/test/good/000.png
   ```
   → Point at the green **NORMAL** banner and the calm blue heatmap.

3. **Now show a defective part.**
   ```bash
   python inspect_image.py data/raw/anomaly/mvtec_ad/metal_nut/test/bent/000.png
   ```
   → Point at the red **DEFECTIVE** banner and the red hotspot that lands exactly
   on the bent region. "The model localises the defect — not just a yes/no."

4. **Try a scratch (subtler defect).**
   ```bash
   python inspect_image.py data/raw/anomaly/mvtec_ad/metal_nut/test/scratch/007.png
   ```

5. **(Optional) Drag in your own image.** Any `.png`/`.jpg` works — just pass its path.

---

## Sample images to use

The MVTec metal_nut test set is already on disk:

| Type | Path |
|---|---|
| Normal | `data/raw/anomaly/mvtec_ad/metal_nut/test/good/000.png` |
| Bent | `data/raw/anomaly/mvtec_ad/metal_nut/test/bent/000.png` |
| Scratch | `data/raw/anomaly/mvtec_ad/metal_nut/test/scratch/007.png` |
| Color | `data/raw/anomaly/mvtec_ad/metal_nut/test/color/000.png` |
| Flip | `data/raw/anomaly/mvtec_ad/metal_nut/test/flip/000.png` |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Image not found` | Check the path; wrap it in quotes if it has spaces. |
| `No trained anomaly runs found` | Train first: `python -m src.train_anomaly --category metal_nut --device 0` |
| `CUDA ... not available` | Use `--device cpu`, or reinstall the CUDA torch build (see setup). |
| Result doesn't auto-open | It's still saved — open `result_<name>.png` manually. `--out` sets the path. |
| Every image looks red | You're on an old build; the fixed colour scale (vmin/vmax in `inspect_image.py`) is calibrated for the metal_nut model. Re-pull latest. |

---

## Note on the object detector

The object-detection panel uses the **base `yolov8n.pt`** (COCO-trained), so on
metal nuts it may detect nothing or a loosely-matching COCO class — that's
expected. It's wired in to prove the unified pipeline works; a domain-specific
YOLO model (PCB / packaging) is the next training step. The **anomaly detector is
the trained, accurate component** and is what to focus the demo on.
```
