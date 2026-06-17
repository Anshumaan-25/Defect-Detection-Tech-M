# Defect Detection

A general, generic AI-based defect-detection system. Given a static image of a
product, it flags four classes of issue:

| Defect type         | Technique                          | Library        |
| ------------------- | ---------------------------------- | -------------- |
| Missing components  | Object detection                   | `ultralytics` (YOLO) |
| Damaged packaging   | Object detection                   | `ultralytics` (YOLO) |
| Surface defects     | Anomaly detection (MVTec AD)       | `anomalib`     |
| Wrong labels        | OCR read + compare vs. expected    | `easyocr`      |

Training runs on **Google Colab** (free GPU); the trained model runs **local
inference on a standard laptop**. The inference layer is kept UI-agnostic so it
can later be wrapped in **Streamlit** or **Gradio** with no changes to the core.

## Project layout

```
Defect Detection/
├── configs/
│   └── datasets.yaml          # registry: what to download + from where
├── data/
│   ├── raw/{yolo,anomaly,ocr} # untouched downloads, per modality
│   └── processed/             # normalized / ready-to-train
├── models/
│   ├── yolo/                  # trained detector weights (.pt)
│   └── anomaly/               # trained anomaly model checkpoints
├── notebooks/                 # Colab training notebooks
├── src/
│   ├── config.py              # paths + dataset registry loading
│   ├── dataset_loader.py      # download & organize datasets  ← start here
│   ├── train_yolo.py          # (next) train the YOLO detector
│   └── inference_pipeline.py  # (next) unified local inference
├── requirements.txt
├── .env.example               # copy to .env, add ROBOFLOW_API_KEY
└── .gitignore
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # then paste your Roboflow key
```

## Getting the data

1. Open each dataset on [Roboflow Universe](https://universe.roboflow.com),
   grab its `workspace` / `project` / `version`, and paste them into
   `configs/datasets.yaml`. Flip `enabled: true`.
2. Run the loader:

   ```bash
   python -m src.dataset_loader --list        # show registry entries
   python -m src.dataset_loader --all         # download everything enabled
   python -m src.dataset_loader --name pcb_defects   # just one
   ```

Datasets land under `data/raw/<modality>/<name>/`. MVTec AD is license-gated —
see the note in `configs/datasets.yaml`.

## Roadmap

- [x] Project scaffold + dataset loader
- [ ] `train_yolo.py` — Colab training entry point
- [ ] Anomaly training (anomalib / PatchCore on MVTec AD)
- [ ] `inference_pipeline.py` — combine all four detectors behind one API
- [ ] Streamlit / Gradio demo app
