# Presentation Runbook — Live Demo

How to run and present the defect-detection demo smoothly. Keep this open during the demo.

---

## Easiest way to start it

**Double-click `run_demo.bat`** in the project folder. A window opens, loads the models
(~15–20 s), and prints the URL. No terminal typing needed.

To stop it afterwards: close that window (or press `Ctrl+C`).

## ✅ Pre-demo checklist (2 minutes before)

1. **🔌 Plug in the laptop.** On battery, Windows disables the GPU to save power. On AC power it's solid.
2. **Double-click `run_demo.bat`** and wait for `Running on local URL`.
3. **Open http://127.0.0.1:7860** in a browser and leave the tab open — so there's no loading wait while sir is watching.
   - To let others open it on the same WiFi: `http://<your-laptop-IP>:7860` (find the IP with `ipconfig`).

## 🖥️ The live demo — all three modes

Each mode has built-in **Example** thumbnails (click to load), or upload your own image.

### 1. Surface defects (metal nut) — the strongest result
- Click the **good nut** example → **Inspect** → 🟢 **NORMAL**, calm blue heatmap.
- Click the **bent nut** example → **Inspect** → 🔴 **DEFECTIVE**, red hotspot lands exactly on the defect.
- **Say:** *"This model trained only on good nuts — it never saw a defect — yet flags them at 99.5% AUROC, and shows where the defect is."*

### 2. PCB defects
- Click the **PCB** example → **Inspect** → bounding boxes with class + confidence (e.g. *Missing_Hole 0.82*).
- **Say:** *"A YOLOv8 detector trained on 6 PCB defect types — 0.978 mAP50."*

### 3. Label check (OCR) — real-world example
- Click the **real PCB** example (loads `samples/OCR_test-E3330BM.jpg` with expected **`ELEC-1`** pre-filled) → **Inspect** → 🟢 **LABEL OK**.
- Change expected text to **`ELEC-2`** → **Inspect** → 🔴 **WRONG LABEL**.
- **Say:** *"EasyOCR reads the board's model marking and verifies it against the expected value — no training needed."*
- ⚠️ Use **`ELEC-1`** (the clean printed marking). Do **not** use the etched batch code `E3330BM` — OCR misreads its stylized font.

## 📊 Headline numbers

| Defect type | Technique | Result |
|---|---|---|
| Surface defects | PatchCore | **AUROC 0.9946** |
| PCB / missing parts | YOLOv8 | **mAP50 0.978** |
| Wrong labels | EasyOCR | validated ✅ |

**Framing:** *"One generic, modular pipeline — three AI techniques, each matched to a defect type. Trained locally, runs on a laptop, scales to the cloud."*

## 🛟 If something goes wrong (backup plan)

- **App won't load / browser issue** → use the CLI fallbacks that save annotated images:
  - `python inspect_image.py data/raw/anomaly/mvtec_ad/metal_nut/test/bent/000.png`
  - `python demo_yolo.py`
- **GPU dropped (was on battery)** → plug in, close the window, double-click `run_demo.bat` again.
- **Total failure** → the CLI tools above save PNGs you can open directly. Consider generating a few beforehand as static backups.

> For the deeper "what/how/why" Q&A, see **`PROJECT_BRIEFING.md`**.
