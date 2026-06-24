"""app.py -- Gradio demo UI for the unified defect-detection pipeline.

A single web app that wraps ``DefectInspector`` behind an upload-an-image form.
Because the trained models are domain-specific, the UI uses a **mode selector**
so each detector runs only where it's meaningful:

    * Surface defects (metal nut)  -> PatchCore anomaly heatmap + NORMAL/DEFECTIVE
    * PCB defects                  -> YOLO boxes + per-class detection table
    * Label check (OCR)            -> EasyOCR text read + WRONG-LABEL verdict

Run it:
    python app.py            # local URL + a temporary public share link

Then open the printed http://127.0.0.1:7860 (and the *.gradio.live link to share).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from src.inference_pipeline import DefectInspector
# Reuse the tested visualisation helpers from the CLI demo tool.
from inspect_image import normalize_map, build_overlay_panel, build_original_panel

# ── Mode labels ──────────────────────────────────────────────────────────────
MODE_SURFACE = "Surface defects (metal nut)"
MODE_PCB = "PCB defects"
MODE_LABEL = "Label check (OCR)"
MODES = [MODE_SURFACE, MODE_PCB, MODE_LABEL]

PCB_WEIGHTS = PROJECT_ROOT / "models/yolo/pcb_defects_yolo/weights/best.pt"

# ── Load all models once at startup ──────────────────────────────────────────
# YOLO uses the trained PCB detector so "PCB defects" mode is meaningful;
# PatchCore auto-loads the latest anomaly run (metal_nut); EasyOCR for labels.
print("Loading models (this takes a few seconds)…")
INSPECTOR = DefectInspector(
    yolo_weights=str(PCB_WEIGHTS) if PCB_WEIGHTS.exists() else None,
    device="auto",
    enable_ocr=True,
)
print(f"Models loaded on device: {INSPECTOR.device}")


# ── Inference callback ───────────────────────────────────────────────────────
def inspect(mode: str, image, expected_text: str):
    """Run the relevant detector for the chosen mode; return (image, verdict, details)."""
    if image is None:
        return None, "### ⚠️ Please upload an image first.", ""

    pil = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
    result = INSPECTOR.inspect(pil, expected_text=(expected_text or None))

    if result["errors"]:
        # Surface any per-detector failure but keep going with whatever succeeded.
        err = "  \n".join(f"⚠️ {e}" for e in result["errors"])
    else:
        err = ""

    if mode == MODE_SURFACE:
        return _surface_view(pil, result, err)
    if mode == MODE_PCB:
        return _pcb_view(pil, result, err)
    return _label_view(pil, result, expected_text, err)


def _surface_view(pil, result, err):
    amap = result["anomaly_map"]
    score = result["anomaly_score"]
    if amap is None:
        return pil, "### ⚠️ Anomaly model unavailable", err
    overlay = build_overlay_panel(pil, normalize_map(amap))
    is_anom = bool(result["is_anomalous"])
    verdict = "### 🔴 DEFECTIVE" if is_anom else "### 🟢 NORMAL"
    details = f"**Anomaly score:** {score:.4f}  \n*(higher = more anomalous; red regions on the heatmap show where)*"
    return overlay, verdict, (details + (f"  \n\n{err}" if err else ""))


def _pcb_view(pil, result, err):
    dets = result["detections"]
    annotated = build_original_panel(pil, dets)
    n = len(dets)
    verdict = f"### 🔴 {n} defect(s) detected" if n else "### 🟢 No defects detected"
    if dets:
        rows = "\n".join(f"| {d['label']} | {d['confidence']:.2f} |" for d in dets)
        details = "| Defect class | Confidence |\n|---|---|\n" + rows
    else:
        details = "_No defects above the confidence threshold._"
    return annotated, verdict, (details + (f"  \n\n{err}" if err else ""))


def _label_view(pil, result, expected_text, err):
    ocr = result["ocr"]
    if ocr is None:
        return pil, "### ⚠️ OCR unavailable", err
    text_found = ocr["text_found"]
    if expected_text:
        ok = ocr["label_ok"]
        verdict = "### 🟢 LABEL OK" if ok else "### 🔴 WRONG LABEL"
    else:
        verdict = "### ℹ️ Text read (enter expected text to verify)"
    details = f"**Text found:** {', '.join(text_found) if text_found else '(none)'}"
    if expected_text:
        details += f"  \n**Expected:** {expected_text}"
    return pil, verdict, (details + (f"  \n\n{err}" if err else ""))


# ── Example images (only those that exist on disk) ───────────────────────────
def _examples():
    candidates = [
        [MODE_SURFACE, "data/raw/anomaly/mvtec_ad/metal_nut/test/good/000.png", ""],
        [MODE_SURFACE, "data/raw/anomaly/mvtec_ad/metal_nut/test/bent/000.png", ""],
    ]
    pcb_dir = PROJECT_ROOT / "data/raw/yolo/pcb_defects/test/images"
    if pcb_dir.is_dir():
        first = next((p for p in sorted(pcb_dir.iterdir())
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), None)
        if first:
            candidates.append([MODE_PCB, str(first), ""])
    # Label-check example: real PCB board marking (verify "ELEC-1").
    # One click loads image + expected text.
    candidates.append([MODE_LABEL, "samples/OCR_test-E3330BM.jpg", "ELEC-1"])
    return [c for c in candidates if Path(c[1]).exists()]


# ── Build the UI ─────────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Defect Detection Demo") as demo:
        gr.Markdown(
            "# 🔍 Defect Detection — Live Demo\n"
            "Pick an **inspection mode**, upload an image, and click **Inspect**. "
            "Models: PatchCore (surface anomalies) · YOLOv8 (PCB defects) · EasyOCR (labels)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(MODES, value=MODE_SURFACE, label="Inspection mode")
                image = gr.Image(label="Upload image", type="pil", height=300)
                expected = gr.Textbox(
                    label="Expected label text (used in Label check mode)",
                    placeholder="e.g. ELEC-1",
                )
                btn = gr.Button("Inspect", variant="primary")
            with gr.Column(scale=1):
                out_img = gr.Image(label="Result", height=300)
                out_verdict = gr.Markdown()
                out_details = gr.Markdown()

        examples = _examples()
        if examples:
            gr.Examples(examples=examples, inputs=[mode, image, expected],
                        label="Example images")

        btn.click(inspect, [mode, image, expected], [out_img, out_verdict, out_details])

    return demo


if __name__ == "__main__":
    # server_name="0.0.0.0" makes the app reachable from other devices on the same
    # network (http://<this-laptop-ip>:7860) — reliable for in-room demo sharing.
    # share=True also attempts a public gradio.live link (may be blocked by AV/firewall).
    build_ui().launch(
        server_name="0.0.0.0",
        share=True,
        theme=gr.themes.Soft(),
    )
