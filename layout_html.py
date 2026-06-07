#!/usr/bin/env python3
"""
layout_html.py - Per-block OCR and layout-preserving HTML assembly.

For a page image that has been analyzed by imgextract.py:
  1. Crop each text block from the original page image
  2. Extract text from each cropped text block using Tesseract OCR
  3. Crop each graphical object from the original page image
  4. Assemble all text blocks and image crops into a single HTML document
     with absolute positioning that preserves the original layout exactly.

Usage (imported from imgextract.py):
    from layout_html import assemble_layout_html
    html = assemble_layout_html(img, h, w, graphical_regions, text_blocks)
"""

import sys
import base64
import io

import pytesseract

import cv2
import numpy as np

# -- OCR configuration --
TESSERACT_LANG = "eng"
TESSERACT_PSM = "6"    # Assume a single uniform block of text
TESSERACT_OEM = "1"    # Tesseract 4+ LSTM only
TESSERACT_CMD = "/opt/metis/el8/contrib/tesseract/tesseract-latest-gcc-12.3.0/bin/tesseract"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def ocr_with_ollama(block_img: np.ndarray, model: str) -> str:
    """OCR a single text-block image using an Ollama vision model.

    Parameters
    ----------
    block_img : np.ndarray
        Cropped BGR numpy array of a text block region.
    model : str
        Ollama model name (e.g. ``qwen3-vl:8b``).

    Returns
    -------
    str
        Extracted text content (raw, no markdown).
    """
    import ollama

    rgb = cv2.cvtColor(block_img, cv2.COLOR_BGR2RGB)
    from PIL import Image
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    prompt = (
        "Extract ALL text from this image block exactly as it appears. "
        "Preserve formatting, special characters, and structure. "
        "Output ONLY the raw text — no explanations, no code fences."
    )

    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [img_bytes],
        }],
        options={"temperature": 0},
    )
    raw = response["message"]["content"].strip()

    # Strip markdown code fences if the model wraps output
    for fence in ["```html", "```python", "```"]:
        if raw.startswith(fence):
            raw = raw[len(fence):]
    raw = raw.strip()
    if raw.startswith("`"):
        raw = raw[1:]
    raw = raw.strip()
    while raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    # ---- Aggressive preamble/notes removal ----
    # The model may echo the prompt back or add its own preamble/notes.
    # Strategy: split into paragraphs, find the first paragraph that looks like
    # actual OCR content (doesn't match any meta-commentary pattern), and drop
    # everything before it. Then drop any remaining paragraphs that are pure notes.
    import re

    # Common meta-commentary patterns that models like to add
    _META_PATTERNS = [
        r'(Here\s+(is|are|was)|Here\'s\s+)',
        r'The\s+text\s+(in\s+this\s+image\s+of\s+|from\s+this\s+image\s+|of\s+the\s+page\s+)',
        r'The\s+extracted\s+text\s+',
        r'This\s+is\s+(the\s+text\s+|a\s+transcription\s+)',
        r'Below\s+is\s+(the\s+)?(extracted\s+)?text\s+',
        r'Extracted\s+text\s*:',
        r'OCR\s+result\s*:',
        r'Answer\s*:',
        r'The\s+OCR\s+result\s+',
        r'I\s+can\s+see\s+the\s+following\s+text\s+',
        r'From\s+the\s+image\s+I\s+can\s+see\s+',
        r'This\s+OCR\s+was\s+',
        r'Based\s+on\s+the\s+image\s+',
        r'Note(:|\s+|:?\s+)',
        r'Additional\s+note',
        r'Side\s+note',
        r'Remark\s*:',
        r'Disclaimer',
        r'For\s+your\s+information',
        r'Important\s+note',
        r'Please\s+note',
        r'NB\s*:',
        r'The\s+image\s+shows',
        r'This\s+is\s+an\s+image\s+of\s+',
        r'I\s+can\s+see\s+',
        r'Looking\s+at\s+this\s+image',
        r'From\s+the\s+image',
        r'The\s+content\s+of\s+the\s+image',
        r'This\s+extracted\s+text',
        r'In\s+this\s+image',
    ]
    _META_RE = re.compile('|'.join(_META_PATTERNS), re.IGNORECASE)

    # If the model echoed the prompt, remove the entire prompt block
    prompt_sentences = [
        r'Extract\s+ALL\s+text\s+from\s+this\s+image\s+block',
        r'Preserve\s+formatting,\s+special\s+characters,\s+and\s+structure\.',
        r'Output\s+ONLY\s+the\s+raw\s+text',
    ]
    for sent in prompt_sentences:
        raw = re.sub(r'[\s\n]*' + sent + r'[.,]?\s*\n?', '\n', raw, flags=re.IGNORECASE)

    # Now strip all paragraphs/lines that are meta-commentary
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', raw) if p.strip()]
    if paragraphs:
        # Drop leading paragraphs that are meta-commentary
        content_start = 0
        for i, p in enumerate(paragraphs):
            if _META_RE.match(p):
                content_start = i + 1
            else:
                break

        # Drop trailing paragraphs that are meta-commentary
        content_end = len(paragraphs)
        for i in range(len(paragraphs) - 1, content_start - 1, -1):
            if _META_RE.match(paragraphs[i]):
                content_end = i
            else:
                break

        # Drop any middle paragraphs that are pure meta-commentary
        paragraphs = [p for p in paragraphs[content_start:content_end]
                      if not _META_RE.match(p)]

        raw = '\n\n'.join(paragraphs)

    # Also strip any remaining meta-commentary lines within paragraphs
    lines = raw.split('\n')
    lines = [l for l in lines if not _META_RE.match(l.strip())]
    raw = '\n'.join(lines)

    # Final sweep: remove any remaining "Note:" anywhere
    raw = re.sub(r'Note[:\s][^\n]*\n?', '\n', raw, flags=re.IGNORECASE)
    return raw.strip()


def ocr_text_block(block_img: np.ndarray, model: str = None) -> str:
    """OCR a single text-block image and return extracted text.

    Uses the Ollama vision model when *model* is provided;
    falls back to Tesseract otherwise.

    Parameters
    ----------
    block_img : np.ndarray
        Cropped BGR numpy array of a text block region.
    model : str, optional
        Ollama model name (e.g. ``qwen3-vl:8b``).
        When *None*, Tesseract is used.

    Returns
    -------
    str
        Extracted text content (raw, no markdown).
    """
    if model is not None:
        return ocr_with_ollama(block_img, model)

    # --- Tesseract path (unchanged) ---
    rgb = cv2.cvtColor(block_img, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image for pytesseract
    from PIL import Image
    img = Image.fromarray(rgb)

    # Binarize to improve OCR on scanned pages
    gray = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_thresh = Image.fromarray(thresh)

    ocr_config = (
        f"--psm {TESSERACT_PSM} "
        f"--oem {TESSERACT_OEM}"
    )
    text = pytesseract.image_to_string(
        img_thresh,
        lang=TESSERACT_LANG,
        config=ocr_config,
    )
    return text.strip()


def img_to_data_uri(arr: np.ndarray, fmt: str = "png") -> str:
    """Encode a numpy image array as a base64 data URI."""
    _, buf = cv2.imencode(f".{fmt}", arr)
    encoded = base64.b64encode(buf).decode("utf-8")
    mime = "image/png" if fmt == "png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def sanitize_html(text: str) -> str:
    """Escape HTML special characters in text content."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\"", "&quot;")
    )


def assemble_layout_html(
    img: np.ndarray,
    img_height: int,
    img_width: int,
    graphical_regions: list,
    text_blocks: list,
    model: str = None,
) -> str:
    """Build a layout-preserving HTML document from detected regions.

    Every text block and graphical object is placed at its original position
    using CSS absolute positioning inside a page-sized relative container.

    Parameters
    ----------
    img : np.ndarray
        Original BGR page image.
    img_height, img_width : int
        Page dimensions in pixels.
    graphical_regions : list[dict]
        Graphical object regions from imgextract.py (each has 'bbox' and optionally 'contour').
    text_blocks : list[dict]
        Text block regions from imgextract.py (each has 'bbox').
    model : str, optional
        Ollama model name for OCR. When None, Tesseract is used.

    Returns
    -------
    str
        Complete HTML document as a string.
    """
    # -- 1. Crop and encode all graphical objects (with page boundary clamping) --
    graph_items = []
    for i, r in enumerate(graphical_regions):
        x, y, bw, bh = r["bbox"]
        # Clamp to page boundaries
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        # Clamp dimensions to not exceed page edges
        if x + bw > img_width:
            bw = max(1, img_width - x)
        if y + bh > img_height:
            bh = max(1, img_height - y)
        crop = img[y:y + bh, x:x + bw]
        if crop.size == 0:
            continue
        data_uri = img_to_data_uri(crop, fmt="png")
        graph_items.append({
            "x": x, "y": y, "w": bw, "h": bh,
            "src": data_uri,
        })

    # -- 2. OCR each text block (with page boundary clamping) --
    text_items = []
    for i, tb in enumerate(text_blocks):
        x, y, bw, bh = tb["bbox"]
        # Clamp to page boundaries
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        # Clamp dimensions to not exceed page edges
        if x + bw > img_width:
            bw = max(1, img_width - x)
        if y + bh > img_height:
            bh = max(1, img_height - y)
        crop = img[y:y + bh, x:x + bw]
        if crop.size == 0:
            continue
        text = ocr_text_block(crop, model=model)
        text_items.append({
            "x": x, "y": y, "w": bw, "h": bh,
            "text": sanitize_html(text),
        })

    # -- 3. Build HTML --
    lines = []
    lines.append("<!DOCTYPE html>")
    lines.append("<html lang=\"en\">")
    lines.append("<head>")
    lines.append("<meta charset=\"UTF-8\">")
    lines.append("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">")
    lines.append("<style>")
    lines.append(f"  body {{ margin: 0; padding: 0; background: #f5f5f5; }}")
    lines.append("  .page {")
    lines.append(f"    position: relative;")
    lines.append(f"    width: {img_width}px;")
    lines.append(f"    height: {img_height}px;")
    lines.append("    overflow: hidden;")
    lines.append("    background: #fff;")
    lines.append("    margin: 0 auto;")
    lines.append("  }")
    lines.append("  .text-block {")
    lines.append("    position: absolute;")
    lines.append("    padding: 0;")
    lines.append("    overflow: visible;")
    lines.append("    white-space: pre-wrap;")
    lines.append("    font-family: 'Segoe UI', 'DejaVu Sans', Arial, sans-serif;")
    lines.append("    font-size: 14px;")
    lines.append("    color: #111;")
    lines.append("    line-height: 1.3;")
    lines.append("    letter-spacing: 0.02em;")
    lines.append("  }")
    lines.append("  .graph-block {")
    lines.append("    z-index: 1;")
    lines.append("    position: absolute;")
    lines.append("    overflow: visible;")
    lines.append("  }")
    lines.append("  .graph-block img {")
    lines.append("    display: block;")
    lines.append("    image-rendering: auto;")
    lines.append("  }")
    lines.append("</style>")
    lines.append("</head>")
    lines.append("<body>")
    lines.append("  <div class=\"page\">")

    # Merge graphical_regions and text_blocks into a single list, sorted by reading order
    # (top-to-bottom by center-y, left-to-right by x)
    all_items = []
    for item in text_items:
        item["_type"] = "text"
        all_items.append(item)
    for item in graph_items:
        item["_type"] = "graph"
        all_items.append(item)

    # Sort in reading order
    all_items.sort(key=lambda it: (it["y"] + it["h"] / 2, it["x"]))

    for item in all_items:
        cls = "text-block" if item["_type"] == "text" else "graph-block"
        left = item["x"]
        top = item["y"]
        width = item["w"]
        height = item["h"]

        if item["_type"] == "text":
            lines.append(
                f'    <div class="{cls}" '
                f'style="left:{left}px;top:{top}px;width:{width}px;height:{height}px;">'
                f'{item["text"]}'
                f"</div>"
            )
        else:
            lines.append(
                f'    <div class="{cls}" '
                f'style="left:{left}px;top:{top}px;width:{width}px;height:{height}px;">'
                f'<img src="{item["src"]}" style="width:{width}px;height:{height}px;">'
                f"</div>"
            )

    lines.append("  </div>")
    lines.append("</body>")
    lines.append("</html>")

    return "\n".join(lines) + "\n"
