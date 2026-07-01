# Project Briefing — Defect Detection

A complete reference for understanding and explaining the pipeline (e.g. in a review/Q&A).

---

## 1. The one-line pitch

> A generic, AI-based visual defect-detection system: you give it a product image, and it flags four kinds of defects — **missing components, surface defects, wrong labels, and damaged packaging** — using three different AI techniques, each matched to the kind of defect.

## 2. The four defect types → technique mapping

| Defect type | Technique | Model / Tool | Status |
|---|---|---|---|
| Surface defects (scratches, dents) | **Anomaly detection** | PatchCore (anomalib) | ✅ Trained — AUROC 0.9946 |
| Missing components (PCB) | **Object detection** | YOLOv8 (ultralytics) | ✅ Trained — mAP50 0.978 |
| Wrong labels | **OCR + text compare** | EasyOCR | ✅ Working (no training needed) |
| Damaged packaging | **Object detection** | YOLOv8 (same family) | ✅ Trained — mAP50 0.810 |

*(All four defect types now have a working, trained detector as of the Phase 2 update — see `DEVELOPMENT_JOURNEY.md` §11 for the full story, including a dataset-selection dead end worth knowing about.)*

**Why three different techniques?** Each defect is a fundamentally different problem:

- **Surface defects** are rare and unpredictable — you can't collect labelled examples of every possible scratch. So we use *anomaly detection*: train only on **good** parts, flag anything that deviates.
- **Missing components** need *localization* ("is this part present, and where?") — exactly what object detection (bounding boxes) does.
- **Wrong labels** just need text *read and compared* — a solved problem, so we use pre-trained OCR, no training required.

## 3. Datasets used

**A) MVTec AD — "Metal Nut" category** (surface-defect / anomaly model)
- Industry-standard anomaly-detection benchmark from MVTec (research-license gated — downloaded manually as an archive).
- Layout: `train/good` (~220 **defect-free** images only), `test/` with good + 4 defect types (**bent, color, flip, scratch**), plus pixel-level `ground_truth` masks for evaluation.

**B) PCB Defect Detection (Ultra)** — from **Roboflow Universe** (YOLO model)
- Workspace `cnn-pcb-defect-detection`, project `pcb-defect-detection-ultra`, version 6 (auto-downloaded via the Roboflow API).
- **6,333 train / 1,778 validation / 903 test** images.
- **6 defect classes:** Missing_Hole, MouseBite, Open_Circuit, Short_Circuit, Spur, Spurious_Cooper *("Cooper" is the dataset's own spelling of "Copper").*

**C) OCR** — *no dataset.* EasyOCR ships pre-trained; the wrong-label logic was validated on a real PCB image (`samples/OCR_test-E3330BM.jpg`) by verifying the board's printed model marking `ELEC-1`.

## 4. Tools & tech stack

| Tool | Version | Role |
|---|---|---|
| Python | 3.13 | Language |
| PyTorch | 2.11.0 **+cu128** | Deep-learning engine (CUDA 12.8 GPU build) |
| anomalib | 2.5 | PatchCore anomaly detection + training |
| ultralytics | 8.4 | YOLOv8 object detection + training |
| EasyOCR | 1.7.2 | Text reading for labels |
| Gradio | 6.19 | Web demo UI |
| Roboflow | — | Dataset download |
| Supporting | — | timm (backbones), Lightning (training loop), Pillow/OpenCV/NumPy (images), PyYAML + python-dotenv (config) |

## 5. How each model works (the "how")

**PatchCore (surface defects)** — a *memory-bank* method:
1. A ResNet18 backbone (pre-trained on ImageNet) extracts mid-level features (layers 2 & 3) from **only good** training images.
2. A representative 10% subset ("coreset") of those feature patches is stored in memory.
3. At test time, each patch of a new image is compared to the nearest stored "normal" patch — far away ⇒ anomalous.
- Key point: **no defect labels, no gradient training** — it "trains" in a single pass (~5.5 min). That's why it only needs good samples.

**YOLOv8 (missing components / PCB defects)** — a single-pass CNN that predicts bounding boxes + class in one shot. We **transfer-learned** from the COCO-pretrained `yolov8n` ("nano", 3M params) onto the 6 PCB classes (~12 min, 5 epochs).

**EasyOCR (labels)** — pre-trained text detection + recognition. It reads the text; our code normalizes it (strips case/spaces/punctuation) and checks whether the **expected** label string is present → `label_ok` true/false.

## 6. Training details & results

| | PatchCore (anomaly) | YOLOv8 (PCB) |
|---|---|---|
| Backbone / model | ResNet18 | yolov8n |
| Key settings | image 256, batch 8, coreset 0.1 | imgsz 640, batch 16, 5 epochs |
| Train time | ~5.5 min | ~12.4 min |
| **Results** | **Image AUROC 0.9946**, F1 0.9785, pixel AUROC 0.9835 | **mAP50 0.978**, mAP50-95 0.672 |
| Inference speed | tens of ms/image | **4.3 ms/image** |

Both were trained **locally on the laptop GPU** — not pre-downloaded models.

## 7. How it all runs together

- **`src/inference_pipeline.py` → `DefectInspector`** is the unified brain. It loads all three models once; `inspect(image, expected_text)` runs them independently (one failing never stops the others) and returns a single dict: YOLO detections + anomaly score/heatmap/verdict + OCR text/label-check.
- **`app.py`** is the Gradio web UI wrapping that, with a mode selector so each model is shown where it's relevant.
- Config is centralized (`src/config.py`); datasets are registry-driven (`configs/datasets.yaml`) — so it runs identically on a laptop or on Google Colab.

## 8. Hardware & engineering story

- **NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM**, 16 GB RAM, Windows 11.
- Everything was tuned **lightweight for 4 GB** (small backbone, small batches) but built to **scale up on Colab** with no code changes.
- Problems solved along the way: a CPU-only PyTorch replaced with the CUDA build; the GPU switching off on low battery (Optimus power-saving); a RAM-exhaustion crash from too many data-loader workers. All documented in `DEVELOPMENT_JOURNEY.md`.

## 9. Honest limitations

- The trained models are **domain-specific**: the anomaly model knows metal nuts, the YOLO model knows PCBs. A new product would need retraining — by design, this is a *generic framework*, not one universal model.
- Training was **smoke-test scale** (5 epochs / small backbone). The numbers are strong but on a relatively clean dataset; a production model would train longer on Colab.
- **Damaged packaging** isn't separately trained (needs a packaging dataset).
- OCR is validated on a **synthetic** label, not yet real product-label photos.

## 10. Likely questions — with answers

- **"What dataset did you use?"** → MVTec AD (Metal Nut) for surface defects; a Roboflow PCB-defect dataset (6 classes, ~9k images) for object detection; OCR needs none.
- **"Did you train these yourself?"** → Yes, both locally on the RTX 3050 — anomaly ~5.5 min, YOLO ~12 min.
- **"How accurate is it?"** → Surface AUROC 0.9946; PCB mAP50 0.978.
- **"Is it real-time?"** → YOLO ~4 ms/image; near real-time on the GPU.
- **"How does it detect defects it's never seen?"** → PatchCore learns what *normal* looks like and flags deviations — no defect examples needed.
- **"Can it work on our product?"** → Yes — it's modular; retrain the anomaly model on good samples and/or YOLO on labelled examples of that product.
- **"What if there's no GPU?"** → It runs on CPU too, just slower (device is auto-detected).
- **"Why such small models?"** → To fit the 4 GB laptop GPU for development; it scales to bigger models on Colab.
