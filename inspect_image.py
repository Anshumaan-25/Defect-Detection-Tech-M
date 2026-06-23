"""inspect_image.py -- run ANY image through the defect-detection pipeline and
render a neat, presentation-ready result panel.

Point it at a single image; it loads the trained models once, runs inference,
and produces a 3-panel annotated result:

    [ Original + YOLO boxes ]  [ Heatmap Overlay ]  [ Pure Anomaly Heatmap ]

...topped with a header (file name + model) and a bottom VERDICT banner
(green NORMAL / red DEFECTIVE) showing the anomaly score and detection count.
The result is saved next to the input and opened automatically.

Usage
-----
    # Inspect any image (auto GPU, opens the result when done):
    python inspect_image.py path/to/image.png

    # Force CPU, write to a specific file, don't auto-open:
    python inspect_image.py path/to/image.png --device cpu --out result.png --no-open

    # Override the anomaly decision threshold:
    python inspect_image.py path/to/image.png --threshold 0.5

Quick test with a bundled MVTec sample:
    python inspect_image.py data/raw/anomaly/mvtec_ad/metal_nut/test/bent/000.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the Windows console happy with any unicode we print.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure 'src' is importable when run as a plain script from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_pipeline import DefectInspector  # noqa: E402

# ── Layout / palette ─────────────────────────────────────────────────────────
DISPLAY_H    = 440               # each panel is resized to this height
HEADER_H     = 46                # top header bar height
SUBLABEL_H   = 30                # per-panel caption strip height
BANNER_H     = 84                # bottom verdict banner height
GAP          = 6                 # gap between panels
BG           = (15, 15, 18)      # canvas background
SUBLABEL_BG  = (30, 30, 34)
GOOD_COLOR   = (34, 197, 94)     # green
DEFECT_COLOR = (239, 68, 68)     # red
BOX_COLOR    = (250, 204, 21)    # amber for YOLO boxes
OVERLAY_ALPHA = 0.55             # heatmap opacity when blended over the original

# Fixed colour scale for the anomaly heatmap. anomalib normalises maps so the
# decision threshold sits near 0.5: a NORMAL nut's map peaks around ~0.48, while
# defects climb to 0.75-0.82. Using a FIXED range (not per-image min/max) keeps
# brightness comparable across images -- normal images stay cool blue/green and
# real defects light up red, instead of every image being stretched to full red.
HEATMAP_VMIN = 0.30
HEATMAP_VMAX = 0.72


# ── Fonts (fall back to PIL default if Arial isn't present) ──────────────────
def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except (IOError, OSError):
        return ImageFont.load_default()


def _centered(draw: ImageDraw.ImageDraw, text: str, x0: int, w: int, y: int,
              font: ImageFont.FreeTypeFont, fill) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((x0 + (w - tw) / 2, y), text, fill=fill, font=font)


# ── Heatmap rendering ────────────────────────────────────────────────────────
def heatmap_to_rgb(amap: np.ndarray) -> np.ndarray:
    """Map a normalised (H,W) float array in [0,1] to a jet-style uint8 RGB image.

    low -> blue (normal), mid -> green/yellow, high -> red (anomalous).
    """
    norm = np.clip(amap, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def normalize_map(amap: np.ndarray) -> np.ndarray:
    """Scale an anomaly map to [0,1] using the FIXED display range above.

    A fixed scale (rather than per-image min/max) makes heatmap brightness
    comparable between images, so a normal sample reads cool and a defective one
    reads hot -- matching the verdict instead of contradicting it.
    """
    rng = HEATMAP_VMAX - HEATMAP_VMIN
    return np.clip((amap - HEATMAP_VMIN) / rng, 0.0, 1.0).astype(np.float32)


# ── Panel builders ───────────────────────────────────────────────────────────
def _resize_to_display(img: Image.Image) -> Image.Image:
    w, h = img.size
    scale = DISPLAY_H / h
    return img.resize((max(1, int(w * scale)), DISPLAY_H), Image.BILINEAR)


def build_original_panel(img: Image.Image, detections: list[dict]) -> Image.Image:
    """The original image, resized for display, with YOLO boxes drawn on top."""
    disp = _resize_to_display(img.convert("RGB"))
    w0, h0 = img.size
    dw, dh = disp.size
    sx, sy = dw / w0, dh / h0

    draw = ImageDraw.Draw(disp)
    font = _font(13)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        box = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        draw.rectangle(box, outline=BOX_COLOR, width=3)
        tag = f"{det['label']} {det['confidence']:.2f}"
        tb = draw.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([box[0], box[1] - th - 4, box[0] + tw + 6, box[1]], fill=BOX_COLOR)
        draw.text((box[0] + 3, box[1] - th - 3), tag, fill=(0, 0, 0), font=font)
    return disp


def build_overlay_panel(img: Image.Image, amap_norm: np.ndarray) -> Image.Image:
    """Jet heatmap alpha-blended over a desaturated copy of the original."""
    disp = _resize_to_display(img.convert("RGB"))
    dw, dh = disp.size

    heat = Image.fromarray(heatmap_to_rgb(amap_norm)).resize((dw, dh), Image.BILINEAR)
    # Desaturate the base so the colour heatmap reads clearly on top.
    base = disp.convert("L").convert("RGB")
    return Image.blend(base, heat, OVERLAY_ALPHA)


def build_pure_heatmap_panel(img: Image.Image, amap_norm: np.ndarray) -> Image.Image:
    dw, dh = _resize_to_display(img.convert("RGB")).size
    return Image.fromarray(heatmap_to_rgb(amap_norm)).resize((dw, dh), Image.BILINEAR)


def with_caption(panel: Image.Image, caption: str) -> Image.Image:
    """Add a caption strip above a panel."""
    w, h = panel.size
    canvas = Image.new("RGB", (w, h + SUBLABEL_H), SUBLABEL_BG)
    canvas.paste(panel, (0, SUBLABEL_H))
    draw = ImageDraw.Draw(canvas)
    _centered(draw, caption, 0, w, 7, _font(14), (190, 190, 195))
    return canvas


# ── Full result composition ──────────────────────────────────────────────────
def compose_result(img: Image.Image, result: dict, source_name: str) -> Image.Image:
    """Assemble header + 3 captioned panels + verdict banner into one image."""
    has_anomaly = result["anomaly_map"] is not None
    if has_anomaly:
        amap_norm = normalize_map(result["anomaly_map"])
        panels = [
            with_caption(build_original_panel(img, result["detections"]),
                         f"Original  ({len(result['detections'])} object(s) detected)"),
            with_caption(build_overlay_panel(img, amap_norm), "Anomaly Heatmap Overlay"),
            with_caption(build_pure_heatmap_panel(img, amap_norm), "Pure Anomaly Map"),
        ]
    else:
        # Anomaly model failed; still show the original so the demo doesn't break.
        panels = [with_caption(build_original_panel(img, result["detections"]),
                               "Original (anomaly model unavailable)")]

    strip_w = sum(p.size[0] for p in panels) + GAP * (len(panels) - 1)
    strip_h = max(p.size[1] for p in panels)

    total_w = strip_w
    total_h = HEADER_H + strip_h + BANNER_H
    canvas = Image.new("RGB", (total_w, total_h), BG)

    # Panels row.
    x = 0
    for p in panels:
        canvas.paste(p, (x, HEADER_H))
        x += p.size[0] + GAP

    draw = ImageDraw.Draw(canvas)

    # Header.
    draw.rectangle([0, 0, total_w, HEADER_H], fill=(24, 24, 28))
    draw.text((16, 8), f"Defect Inspection  |  {source_name}", fill=(235, 235, 240), font=_font(18))
    draw.text((16, 28), "PatchCore/ResNet18 (anomaly)  +  YOLOv8n (objects)",
              fill=(130, 130, 140), font=_font(11))

    # Verdict banner.
    is_anom = bool(result["is_anomalous"])
    color = DEFECT_COLOR if is_anom else GOOD_COLOR
    by0 = HEADER_H + strip_h
    draw.rectangle([0, by0, total_w, total_h], fill=color)

    verdict = "VERDICT:  DEFECTIVE" if is_anom else "VERDICT:  NORMAL"
    score = result["anomaly_score"]
    score_txt = f"Anomaly score: {score:.4f}" if score is not None else "Anomaly score: n/a"
    _centered(draw, verdict, 0, total_w, by0 + 14, _font(30), "white")
    _centered(draw, score_txt, 0, total_w, by0 + 54, _font(16), "white")

    return canvas


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a single image and render an annotated result panel.")
    parser.add_argument("image", help="path to the image to inspect")
    parser.add_argument("--device", default="auto",
                        help="auto | cpu | cuda | <int gpu index>  (default: auto)")
    parser.add_argument("--out", default=None,
                        help="output PNG path (default: result_<imagename>.png)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override anomaly score threshold for the verdict")
    parser.add_argument("--anomaly-run-dir", default=None,
                        help="specific PatchCore run dir (default: auto-detect latest)")
    parser.add_argument("--yolo-weights", default=None, help="path to YOLO .pt weights")
    parser.add_argument("--no-open", action="store_true",
                        help="don't open the result image when done")
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        return 1

    out_path = Path(args.out) if args.out else PROJECT_ROOT / f"result_{image_path.stem}.png"

    print("\n==================================================")
    print("  Defect Detection -- Single Image Inspection")
    print("==================================================")
    print(f"  Image : {image_path.name}")

    print("  [1/3] Loading models (YOLO + PatchCore)...")
    inspector = DefectInspector(
        anomaly_run_dir=args.anomaly_run_dir,
        yolo_weights=args.yolo_weights,
        device=args.device,
        anomaly_threshold=args.threshold,
        enable_ocr=False,  # this panel visualises anomaly + YOLO only
    )

    print("  [2/3] Running inference...")
    img = Image.open(image_path).convert("RGB")
    result = inspector.inspect(image_path)
    for err in result["errors"]:
        print(f"        [WARNING] {err}")

    score = result["anomaly_score"]
    verdict = "DEFECTIVE" if result["is_anomalous"] else "NORMAL"
    print(f"        Objects detected : {len(result['detections'])}")
    print(f"        Anomaly score    : {score:.4f}" if score is not None else
          "        Anomaly score    : n/a")
    print(f"        VERDICT          : {verdict}")

    print("  [3/3] Rendering result panel...")
    panel = compose_result(img, result, image_path.name)
    panel.save(out_path)
    print(f"        Saved -> {out_path}")
    print("==================================================\n")

    if not args.no_open:
        try:
            os.startfile(str(out_path))  # Windows: opens in the default image viewer
        except (AttributeError, OSError):
            pass  # non-Windows or no viewer; the file is still saved

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
