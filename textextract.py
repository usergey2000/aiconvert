#!/usr/bin/env python3
"""
textextract.py - Extract text from image files (PNG, JPG, etc.) using OpenCV
for preprocessing and a single vision model call for OCR + grammar correction + HTML
formatting, saving output as HTML while preserving the original layout and formatting.

Workflow per image:
  1. OpenCV loads and preprocesses the image (deskew, contrast-enhance, denoise)
  2. qwen3-vl:8b performs OCR, corrects errors, and formats as HTML in one call
  3. Saves pageN_text.html (corrected HTML with full HTML wrapper)

Usage:
    python3 textextract.py image1.png image2.png
    python3 textextract.py *.png
    python3 textextract.py page-*.png -o output_dir/
"""

import sys
import os
import argparse
import json
import base64
import re

import cv2
import numpy as np
import ollama


# =====================================================================
# OpenCV image preprocessing pipeline
# =====================================================================

def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR uint8 numpy array.

    Returns None if the file cannot be read.
    """
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: cannot read image '{path}'", file=sys.stderr)
    return img


def image_to_bgr_bytes(img: np.ndarray, fmt: str = ".png") -> bytes:
    """Encode a numpy image array as bytes in the given format (PNG, JPG, ...)."""
    _, buf = cv2.imencode(fmt, img)
    return buf.tobytes()


def grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def deskew(img: np.ndarray) -> np.ndarray:
    """Detect and correct page skew by finding the dominant edge angle.

    Returns a deskewed image. If no skew is detected, returns the original.
    """
    gray = grayscale(img)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return img

    # Collect angles, ignoring vertical lines (near 90 degrees)
    angles = []
    for rho, theta in lines[:, 0]:
        deg = theta * 180 / np.pi - 90
        if abs(deg) > 3:  # ignore near-vertical lines
            angles.append(deg)

    if not angles:
        return img

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:  # less than half a degree — skip
        return img

    # Rotate to correct skew
    h, w = gray.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Works on grayscale, then converts back to BGR if the input was color.
    """
    gray = grayscale(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    if len(img.shape) == 3:
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced


def denoise(img: np.ndarray) -> np.ndarray:
    """Remove noise while preserving text edges."""
    gray = grayscale(img)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    if len(img.shape) == 3:
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    return denoised


def binarize(img: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding to produce a clean binary image.

    Useful when the final output is intended to be text-only (e.g., saving
    intermediate preprocessed views). We keep the enhanced contrast version
    for sending to the VL model to preserve color cues.
    """
    gray = grayscale(img)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=5
    )
    if len(img.shape) == 3:
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return binary


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Apply the full preprocessing pipeline and return a clean BGR image.

    Pipeline: deskew -> denoise -> CLAHE contrast enhancement.
    """
    img = deskew(img)
    img = denoise(img)
    img = enhance_contrast(img)
    return img


def get_image_preview(img: np.ndarray, max_dim: int = 1200) -> np.ndarray:
    """Downscale the image so that its largest dimension does not exceed max_dim.

    This keeps the VL model's context window from being overwhelmed while
    still providing sufficient text resolution (OCR typically needs ~150 dpi
    line height).
    """
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)  # never upscale
    if scale >= 1.0:
        return img
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# =====================================================================
# Ollama interaction helpers (using the Python library)
# =====================================================================

OLLAMA_TEMPERATURE = 0                       # deterministic output


def extract_and_format_html(img: np.ndarray, model: str) -> str:
    """Use qwen3-vl to extract all text from the image and format as HTML in a single call.

    The image is preprocessed (deskewed, denoised, contrast-enhanced) and
    optionally down-scaled before being sent to the model.

    Returns the corrected HTML string.
    """
    # Preprocess
    processed = preprocess_image(img)

    # Downscale if the image is very large (preserves text legibility)
    preview = get_image_preview(processed, max_dim=1600)

    # Encode to PNG bytes for the ollama API
    img_bytes = image_to_bgr_bytes(preview, fmt=".png")

    prompt = (
        "Extract ALL text from this image exactly as it appears, then output it as HTML.\n\n"
        "Requirements:\n"
        "1. CORRECT any OCR errors in grammar/spelling while preserving all technical terms, model numbers, and specifications.\n"
        "2. PRESERVE the original layout and structure:\n"
        "   - Convert tables into <table>/<tr>/<td> HTML elements with column alignment\n"
        "   - Keep section headers (<h2>, <h3> etc.), lists (<ul>/<li> or <ol>/<li>), and indentation\n"
        "   - Preserve special characters, abbreviations, and acronyms exactly as shown\n"
        "3. Do NOT change the meaning of any technical content.\n"
        "4. Preserve ALL text — do not skip anything, even small or technical content.\n"
        "5. Wrap the entire output in <html>/<head>/<body> tags.\n"
        "6. Use <style> for basic formatting: fonts, borders, table styles.\n\n"
        "OUTPUT: ONLY the complete HTML document. No explanations, no markdown, no code fences."
    )

    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [img_bytes],
        }],
        options={"temperature": OLLAMA_TEMPERATURE},
    )
    return response["message"]["content"]



# =====================================================================
# HTML post-processing
# =====================================================================

def clean_ollama_output(text: str) -> str:
    """Strip markdown code fences and stray whitespace from ollama output."""
    text = text.strip()

    # Remove markdown code fences if present (handle both ``` and ```)
    for fence in ["```html", "```python", "```", "```python"]:
        if text.startswith(fence):
            text = text[len(fence):]
    text = text.strip()
    if text.startswith("`"):
        text = text[1:]
    text = text.strip()
    if text.endswith("`"):
        text = text[:-1]
    while text.endswith("```"):
        text = text[:-3]
    return text.strip()


def ensure_html_completeness(html: str) -> str:
    """Ensure the HTML output has a proper <html> wrapper if missing."""
    stripped = html.strip()
    if not stripped.startswith("<html") and not stripped.startswith("<!DOCTYPE"):
        html = (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "<meta charset=\"UTF-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            "<title>Extracted Text</title>\n"
            "<style>\n"
            "  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; line-height: 1.6; color: #222; }\n"
            "  table { border-collapse: collapse; width: 100%; margin: 12px 0; }\n"
            "  th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }\n"
            "  th { background: #f0f0f0; font-weight: bold; }\n"
            "  h2 { border-bottom: 2px solid #444; padding-bottom: 4px; }\n"
            "  code { background: #f5f5f5; padding: 2px 4px; border-radius: 3px; }\n"
            "</style>\n"
            "</head>\n<body>\n"
        ) + stripped + "\n</body>\n</html>\n"
    return html


# =====================================================================
# Image utility
# =====================================================================

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def is_image(path: str) -> bool:
    """Check if path is a supported image file."""
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS


def image_base(path: str) -> str:
    """Return 'pageN' style base from filename."""
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r'page-(\d+)', base, re.IGNORECASE)
    return f"page{m.group(1)}" if m else base


# =====================================================================
# Main processing
# =====================================================================

def process_image(image_path: str, out_dir: str, verbose: bool, model: str) -> dict:
    """Process a single image: load, preprocess, extract+format as HTML, save."""
    base = image_base(image_path)
    summary = {"image": os.path.basename(image_path), "steps": {}}

    # ---- Step 0: Load and inspect image with OpenCV ----
    img = load_image(image_path)
    if img is None:
        raise RuntimeError(f"Cannot load image: {image_path}")
    h, w = img.shape[:2]
    summary["image_info"] = {"width": w, "height": h, "dtype": str(img.dtype)}

    # Save a preprocessed preview for debugging
    preprocessed = preprocess_image(img)
    preview_path = os.path.join(out_dir, f"{base}_preprocessed.png")
    cv2.imwrite(preview_path, preprocessed)
    summary["steps"]["preprocess"] = {
        "status": "ok",
        "output": "preprocessed.png",
        "preview_path": preview_path,
    }
    if verbose:
        print(f"  Image: {w}x{h}  dtype={img.dtype}")
        print(f"  Preprocessed preview saved to {preview_path}")

    # ---- Step 1: Single call — OCR + correction + HTML formatting via Ollama ----
    if verbose:
        print(f"  [1/2] Extracting text and formatting HTML with {model}...")
    raw_html = extract_and_format_html(img, model)

    # Post-process: strip code fences if the model wraps output in markdown
    corrected_html = clean_ollama_output(raw_html)
    corrected_html = ensure_html_completeness(corrected_html)

    if verbose:
        preview = raw_html[:200].replace("\n", "\\n")
        print(f"  Generated {len(corrected_html)} chars of HTML. Preview: {preview}...")

    # ---- Step 2: Save HTML ----
    html_path = os.path.join(out_dir, f"{base}_text.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(corrected_html)
    summary["steps"]["extract_and_format"] = {
        "status": "ok",
        "output": "text.html",
        "char_count": len(corrected_html),
    }

    if verbose:
        print(f"  [2/2] Saved HTML to {html_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from images using OpenCV + Ollama, correct grammar/spelling, save as HTML."
    )
    parser.add_argument("images", nargs="+", help="Input image files (PNG, JPG, etc.)")
    parser.add_argument("--output", "-o", default=None, help="Output directory (default: same dir as images)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without running")
    parser.add_argument("--model", "-m", default="qwen3-vl:8b", help="Ollama vision model to use (default: qwen3-vl:8b)")
    args = parser.parse_args()

    # Validate input files
    valid_images = []
    for path in args.images:
        if is_image(path):
            valid_images.append(path)
        else:
            print(f"Warning: skipping '{path}' (not a supported image file)")
    if not valid_images:
        print("Error: no valid image files found.", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    out_dir = args.output if args.output else os.path.dirname(os.path.abspath(valid_images[0]))
    os.makedirs(out_dir, exist_ok=True)

    if args.dry_run:
        print(f"Would process {len(valid_images)} image(s):")
        for img in valid_images:
            base = image_base(img)
            print(f"  {img} -> {out_dir}/{base}_text.html")
        return

    # Process each image
    total_summary = {
        "images_processed": len(valid_images),
        "output_dir": out_dir,
        "images": [],
    }

    for img in valid_images:
        print(f"\nProcessing: {img}")
        try:
            result = process_image(img, out_dir, args.verbose, args.model)
            total_summary["images"].append(result)
        except Exception as e:
            print(f"Error processing {img}: {e}", file=sys.stderr)
            total_summary["images"].append({
                "image": os.path.basename(img),
                "steps": {"error": str(e)},
            })

    # Save overall summary
    summary_path = os.path.join(out_dir, "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(total_summary, f, indent=2, ensure_ascii=False)

    success = sum(1 for img in total_summary["images"] if "error" not in img.get("steps", {}))
    print(f"\nDone: {success}/{len(valid_images)} image(s) processed successfully.")
    print(f"Output directory: {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
