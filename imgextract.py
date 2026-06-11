#!/usr/bin/env python3
"""
imgextract.py - Detect graphical objects in a page image and crop them out
in reading order (top-to-bottom, left-to-right within each row).

Graphical objects include: color-coded tables, diagrams, charts, figures,
photos, and illustrations — anything that is NOT pure text.

For each page image it produces:
  pageN_graph_M.png         cropped graphical objects (sorted in reading order)
  pageN_text_regions.png    bounding boxes around text blocks
  pageN_annotated.png       overlay showing all detected regions
  pageN_summary.json        machine-readable layout analysis

Detection strategies (all run, then results merged):
  1. Color saturation analysis   — color-coded tables/regions stand out vs grayscale text
  2. Edge/contour detection      — distinct graphical shapes
  3. Connected-component analysis — grouped white/bright regions
  4. Gradient-based detection    — diagrams/line art with strong directional gradients

A second pass detects text blocks separately.

Usage:
    python3 imgextract.py <input_image> [-o <dir>] [--min-area-pct 0.005] [--debug]
"""

import sys
import os
import argparse
import json
import re
import numpy as np
import cv2
import layout_html


# ===== Helpers =====

def load_image(path: str) -> np.ndarray:
    """Load image as BGR numpy array."""
    img = cv2.imread(path)
    if img is None:
        print(f"Error: cannot read image '{path}'", file=sys.stderr)
        sys.exit(1)
    return img


def image_base(path: str) -> str:
    """Return 'pageN' style base from filename."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r'page-(\d+)', stem, re.IGNORECASE)
    return f"page{m.group(1)}" if m else stem


def box_area(bbox) -> int:
    """Area of (x,y,w,h) box."""
    return bbox[2] * bbox[3]


def containment(bbox_outer, bbox_inner) -> float:
    """Fraction of inner-bbox area contained within outer bbox."""
    x1, y1, w1, h1 = bbox_outer
    x2, y2, w2, h2 = bbox_inner
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    return inter / max(box_area(bbox_inner), 1)


def proximity_gap(bbox1, bbox2) -> float:
    """Normalized gap between two boxes — 0 if overlapping, <1 if close."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    dx = max(0, x2 - (x1 + w1), x1 - (x2 + w2))
    dy = max(0, y2 - (y1 + h1), y1 - (y2 + h2))
    diag = max(w1, h1, w2, h2)
    return (dx + dy) / max(diag, 1)


def iou(box1, box2) -> float:
    """Intersection-over-union."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / max(union, 1)


def boxes_overlap(box1, box2) -> bool:
    """Check if two (x, y, w, h) boxes have a positive-area overlap."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    return (xi2 > xi1) and (yi2 > yi1)


def resolve_text_text_overlaps(blocks: list, overlap_threshold: float = 0.01) -> tuple:
    """Detect and resolve overlaps between text blocks.

    Blocks are processed in reading-order (input order is preserved).
    For each overlapping pair the block with the smaller area is dropped;
    the larger block keeps its position.  Small pixel-level overlaps below
    *overlap_threshold* (fraction of the smaller box's area) are ignored.

    Returns
    -------
    (resolved_blocks, stats)
        resolved_blocks : list of dicts with 'bbox' (list) and optional 'text'
        stats : dict with 'overlaps_found' and 'overlaps_fixed' counts
    """
    stats = {'overlaps_found': 0, 'overlaps_fixed': 0}
    if len(blocks) < 2:
        return blocks, stats

    # Work on mutable copies (list-of-list bboxes + optional text)
    resolved = [{'bbox': list(b['bbox']), 'text': b.get('text', '')} for b in blocks]
    n = len(resolved)
    to_remove: set[int] = set()

    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            box_i = tuple(resolved[i]['bbox'])
            box_j = tuple(resolved[j]['bbox'])
            if not boxes_overlap(box_i, box_j):
                continue
            stats['overlaps_found'] += 1

            # Intersection dimensions
            x1, y1, w1, h1 = box_i
            x2, y2, w2, h2 = box_j
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)
            ix = max(0, xi2 - xi1)
            iy = max(0, yi2 - yi1)
            overlap_area = ix * iy
            area_i = w1 * h1
            area_j = w2 * h2
            smaller_area = min(area_i, area_j)
            overlap_frac = overlap_area / max(smaller_area, 1)

            if overlap_frac < overlap_threshold:
                continue

            stats['overlaps_fixed'] += 1

            # Drop the smaller block; keep the larger one
            if area_j <= area_i:
                to_remove.add(j)
            else:
                to_remove.add(i)

    return [b for idx, b in enumerate(resolved) if idx not in to_remove], stats


def resolve_text_graph_overlaps(
    text_blocks: list,
    graph_regions: list,
    overlap_threshold: float = 0.01,
    page_w: int = 4000,
    page_h: int = 4000,
) -> tuple:
    """Leave text blocks untouched — they are processed normally even when
    their bboxes intersect graphical objects."""
    stats = {'overlaps_found': 0, 'overlaps_fixed': 0}
    return text_blocks, stats


def _try_move_vertically(
    x1, y1, w1, h1, x2, y2, w2, h2, page_h: int,
) -> tuple:
    """Try to move a text block vertically (above or below a graph)."""
    # Move below graph
    new_y = y2 + h2 + 4
    if new_y + h1 <= page_h:
        return (x1, new_y, w1, h1)
    # Move above graph
    new_y = max(0, y2 - h1 - 4)
    if new_y >= 0:
        return (x1, new_y, w1, h1)
    # Neither side fits — hide (zero width)
    return (x1, y1, 0, h1)


def _try_move_horizontally(
    x1, y1, w1, h1, x2, y2, w2, h2, page_w: int,
) -> tuple:
    """Try to move a text block horizontally (left or right of a graph)."""
    # Move right of graph
    new_x = x2 + w2 + 4
    if new_x + w1 <= page_w:
        return (new_x, y1, w1, h1)
    # Move left of graph
    new_x = max(0, x2 - w1 - 4)
    if new_x >= 0:
        return (new_x, y1, w1, h1)
    # Neither side fits — hide (zero width)
    return (x1, y1, 0, h1)


# ===== Detection strategies =====

def detect_color_regions(img, sat_thresh: int = 30, max_page_frac: float = 0.4) -> list:
    """Find regions with significant color saturation.

    Rejects regions that cover too much of the page (likely noise from
    arrows or page borders) since those cannot be genuine graphical objects.
    """
    h, w = img.shape[:2]
    total = h * w
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > sat_thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 30 or bh < 30:
            continue
        if bw * bh > total * max_page_frac:
            continue
        regions.append({'type': 'color', 'bbox': (x, y, bw, bh), 'contour': cnt})
    return regions


def detect_edge_regions(img, lower: int = 50, upper: int = 200, min_area: int = 1000) -> list:
    """Find graphical regions via Canny edge detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h = img.shape[0]
    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw / max(bh, 1) > 10 or bh > h * 0.95:
            continue
        regions.append({'type': 'edge', 'bbox': (x, y, bw, bh), 'contour': cnt})
    return regions


def detect_cc_regions(img, min_area: int = 1000) -> list:
    """Find non-text (white/bright) regions via connected-component analysis."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=4)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 30 or bh < 30:
            continue
        regions.append({'type': 'connected_comp', 'bbox': (x, y, bw, bh), 'contour': cnt})
    return regions


def detect_gradient_regions(img, min_area: int = 1000) -> list:
    """Find regions with strong directional gradients (diagrams, line art)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.uint8(255 * np.sqrt(gx ** 2 + gy ** 2) / max(np.sqrt(gx ** 2 + gy ** 2).max(), 1))
    _, thresh = cv2.threshold(mag, 80, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 30 or bh < 30:
            continue
        regions.append({'type': 'gradient', 'bbox': (x, y, bw, bh), 'contour': cnt})
    return regions


# ===== Merge =====

def merge_regions(regions, gap_thresh: float = 0.12, max_frac: float = 0.35):
    """Merge overlapping or nearby regions.

    Constraints:
    - A small region is NOT merged into a much larger enclosing region.
    - A merged region's bbox cannot exceed max_frac of the total image area.
    """
    if not regions:
        return []

    # Compute approximate total image area from bounding boxes
    max_x = max(r['bbox'][0] + r['bbox'][2] for r in regions)
    max_y = max(r['bbox'][1] + r['bbox'][3] for r in regions)
    total_area = max(max_x * 1.05, 1) * max(max_y * 1.05, 1)

    merged_bboxes = []
    merged_types = []

    for i, r in enumerate(regions):
        bbox_i = r['bbox']
        type_i = {r['type']}
        dominated = False

        for j in range(len(merged_bboxes)):
            g = proximity_gap(bbox_i, merged_bboxes[j])
            ov = iou(bbox_i, merged_bboxes[j])

            # Don't merge if one fully contains the other
            contain_ij = containment(bbox_i, merged_bboxes[j])
            contain_ji = containment(merged_bboxes[j], bbox_i)
            if contain_ij > 0.9 or contain_ji > 0.9:
                continue

            if ov > 0.15 or g < gap_thresh:
                x1, y1, w1, h1 = merged_bboxes[j]
                x2, y2, w2, h2 = bbox_i
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                new_area = new_w * new_h

                # Reject merge if new bbox exceeds max_frac of total area
                if new_area > max_frac * total_area:
                    continue

                merged_bboxes[j] = (new_x, new_y, new_w, new_h)
                merged_types[j].update(type_i)
                dominated = True
                break

        if not dominated:
            merged_bboxes.append(bbox_i)
            merged_types.append(type_i)

    return [{'bbox': b, 'sources': sorted(t)} for b, t in zip(merged_bboxes, merged_types)]


# ===== Detect text blocks =====

def detect_text_blocks(img, min_area: int = 2000) -> list:
    """Detect text blocks as connected components of dark pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded = cv2.erode(cv2.dilate(thresh, kernel, iterations=2), kernel, iterations=1)
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 50 or bh < 15:
            continue
        # Allow text near all edges.  Reject only true full-page bboxes
        # (e.g. a giant background rect) — these are filtered later by
        # page_frac_threshold in merge_regions.
        if x < w * 0.02 and (x + bw) > w * 0.98:
            continue
        # Reject regions that span most of the page height — these are likely
        # illustrations, tables, or other graphics misclassified as text blobs.
        # Exception: wide blobs (>= 30% page width) are almost certainly multi-line
        # text columns that were merged by morphological operations. Allow them through.
        fw = bw / max(w, 1)
        if bh > h * 0.15 and fw < 0.30:
            continue
        blocks.append({'bbox': (x, y, bw, bh)})
    return blocks


# ===== Reading order =====

def sort_reading_order(regions):
    """Sort regions top-to-bottom, left-to-right within each row."""
    if not regions:
        return []
    ordered = sorted(regions, key=lambda r: r['bbox'][1] + r['bbox'][3] / 2)
    rows = []
    cur = [ordered[0]]
    bottom = ordered[0]['bbox'][1] + ordered[0]['bbox'][3]
    for r in ordered[1:]:
        cy = r['bbox'][1] + r['bbox'][3] / 2
        if cy <= bottom:
            cur.append(r)
            bottom = max(bottom, r['bbox'][1] + r['bbox'][3])
        else:
            cur.sort(key=lambda r: r['bbox'][0])
            rows.append(cur)
            cur = [r]
            bottom = r['bbox'][1] + r['bbox'][3]
    cur.sort(key=lambda r: r['bbox'][0])
    rows.append(cur)
    return [r for row in rows for r in row]


# ===== Annotation overlay =====

def draw_overlay(img, regions, label, outpath):
    """Draw bounding boxes with colored fills and labels."""
    overlay = img.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
              (0, 128, 255), (128, 255, 0), (200, 100, 50), (50, 100, 200)]
    for i, r in enumerate(regions):
        x, y, bw, bh = r['bbox']
        color = colors[i % len(colors)]
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(overlay, f"{label}:{i + 1}", (x + 4, y + 16),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, overlay)
    cv2.imwrite(outpath, overlay)


# ===== Main =====

def main():
    parser = argparse.ArgumentParser(
        description='Detect and crop graphical objects from a page image in reading order.'
    )
    parser.add_argument('image', help='Input page image (PNG, JPG, etc.)')
    parser.add_argument('--output', '-o', default=None, help='Output directory')
    parser.add_argument('--min-area-pct', type=float, default=0.005,
                        help='Min area fraction for graphical objects (default: 0.005)')
    parser.add_argument('--min-text-area', type=int, default=2000,
                        help='Min area for text blocks (default: 2000)')
    parser.add_argument('--gap-threshold', type=float, default=0.12,
                        help='Proximity gap for merging nearby regions (default: 0.12)')
    parser.add_argument('--saturation-threshold', type=int, default=30,
                        help='Saturation threshold for color detection (default: 30)')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Verbosity level: 0=silent (default), -v=summary, -vv=details')
    parser.add_argument('--debug', action='store_true', help='Save intermediate debug masks')
    parser.add_argument('--ollama-model', '-m', default=None,
                        help='Ollama vision model for OCR (e.g. maternion/LightOnOCR-2:1b). '
                             'Note: Standard qwen3-vl models (2b, 4b, 8b) are unstable for OCR and may crash. '
                             'When given, Ollama is used per text block; otherwise Tesseract is used.')
    parser.add_argument('--tess-lang', default='eng',
                        help='Tesseract language(s) to use (default: eng). See tesseract --list-langs.')
    parser.add_argument('--tess-system', action='store_true',
                        help='Use system tesseract binary instead of Python library for OCR.')
    args = parser.parse_args()

    img = load_image(args.image)
    h, w = img.shape[:2]
    base = image_base(args.image)
    out_dir = args.output or os.path.dirname(os.path.abspath(args.image))
    os.makedirs(out_dir, exist_ok=True)

    min_abs_area = h * w * args.min_area_pct
    page_frac_threshold = 0.50  # reject any bbox covering >50% of the page
    max_object_frac = 0.40  # max area fraction for a single graphical object

    if args.verbose >= 1:
        print(f"Input: {args.image} ({w}x{h}, total area={w*h:,})")
        print(f"Output dir: {out_dir}")
        print(f"Min graphical area: {args.min_area_pct} => {min_abs_area:.0f} px")
        print(f"Max object area: {max_object_frac:.0%} => {h*w*max_object_frac:.0f} px")

    # ---- Detect graphical regions ----
    regions = []

    for name, fn in [('color', lambda: detect_color_regions(img, args.saturation_threshold)),
                     ('edge', lambda: detect_edge_regions(img)),
                     ('cc', lambda: detect_cc_regions(img)),
                     ('gradient', lambda: detect_gradient_regions(img))]:
        result = fn()
        if args.verbose >= 1:
            print(f"  {name:8s} regions: {len(result)}")
        regions.extend(result)

    if args.verbose >= 1:
        print(f"  Total raw regions: {len(regions)}")

    # Pre-filter: remove page-spanning regions BEFORE merging
    total_area = h * w
    pre_merged = []
    for r in regions:
        area = box_area(r['bbox'])
        if area > total_area * page_frac_threshold:
            if args.verbose >= 1:
                print(f"  PRE-FILTER (page-spanning): [{r['bbox'][2]}x{r['bbox'][3]}] {area/total_area:.0%}")
            continue
        if area < min_abs_area:
            if args.verbose >= 1:
                print(f"  PRE-FILTER (area): [{r['bbox'][2]}x{r['bbox'][3]}] {area} < {min_abs_area:.0f}")
            continue
        pre_merged.append(r)

    if args.verbose >= 1:
        print(f"  After pre-filter: {len(pre_merged)} / {len(regions)} regions")

    # Merge nearby regions
    merged = merge_regions(pre_merged, gap_thresh=args.gap_threshold)

    if args.verbose >= 1:
        print(f"  After merging: {len(merged)} regions")
        for i, r in enumerate(merged):
            print(f"    {i}: [{r['bbox'][2]}x{r['bbox'][3]}] area={box_area(r['bbox']):,}  sources={r['sources']}")

    # Post-merge deduplication: remove regions mostly contained in another
    post_merge = []
    for i, ri in enumerate(merged):
        dominated = False
        for j, rj in enumerate(merged):
            if i == j:
                continue
            if box_area(ri['bbox']) > box_area(rj['bbox']):
                continue
            if containment(rj['bbox'], ri['bbox']) > 0.85:
                dominated = True
                break
        if not dominated:
            post_merge.append(ri)

    if args.verbose >= 1:
        print(f"  After dedup: {len(post_merge)} regions")

    # Pre-filter oversized rectangular regions likely to be text paragraphs.
    # Run BEFORE the main filter so they are excluded from both normal and
    # relaxed paths (they would otherwise reach the relaxed fallback).
    wide_pages = []
    narrow_graphical = []
    for r in post_merge:
        x, y, bw, bh = r['bbox']
        fw = bw / max(w, 1)
        left_margin = x / max(w, 1)
        right_margin = (w - (x + bw)) / max(w, 1)
        # Oversized if: wide enough AND close to page edge(s) AND paragraph-like shape.
        # This catches full-width paragraphs that start/end near the page margins.
        is_wide = fw > 0.70 and (left_margin < 0.10 or right_margin < 0.10)
        is_para_like = 1.0 < (bw / max(bh, 1)) < 3.0
        if is_wide and is_para_like:
            wide_pages.append(r)
        else:
            narrow_graphical.append(r)

    if args.verbose >= 1 and wide_pages:
        for r in wide_pages:
            x, y, bw, bh = r['bbox']
            left_margin = x / max(w, 1)
            right_margin = (w - (x + bw)) / max(w, 1)
            print(f"  FILTER (oversized): [{bw}x{bh}] {fw:.0%} width  l={left_margin:.0%} r={right_margin:.0%} → likely text")

    if args.verbose >= 1:
        print(f"  Wide-rect rejected: {len(wide_pages)}, passing to filter: {len(narrow_graphical)}")

    # ---- Detect text blocks (needed for post-filter after both normal/relaxed paths) ----
    text_blocks = detect_text_blocks(img, min_area=args.min_text_area)
    if args.verbose >= 1:
        print(f"Detected {len(text_blocks)} text block(s).")

    # Filter to final graphical objects (from narrow_graphical = non-wide-rect candidates)
    graphical = []
    for r in narrow_graphical:
        x, y, bw, bh = r['bbox']
        area = box_area(r['bbox'])
        if area < min_abs_area:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.15 or aspect > 10:
            continue
        if area > total_area * page_frac_threshold:
            continue
        if area > total_area * max_object_frac:
            continue
        margin = 0.02
        if y < h * margin or (y + bh) > h * (1 - margin):
            continue
        if x < w * margin or (x + bw) > w * (1 - margin):
            continue
        # Reject sparse blobs — bounding boxes full of scattered dark pixels on
        # white background are text (letter edges from gradient detection, or
        # dense columns of chars), not actual graphical objects. True graphics
        # (photos, diagrams, charts) have dark content packed densely (>~18% of bbox).
        roi = img[y:y + bh, x:x + bw]
        if roi.size > 0:
            arr = roi.astype(np.float64)
            mean_gray = np.mean(arr)
            dark_pct = np.sum(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y:y+bh, x:x+bw] < 180) / roi.size * 100
            bright_pct = np.sum(np.all(arr > 230, axis=2)) / arr.size * 100
            fw = bw / max(w, 1)
            # Wide-bright: text columns spanning most of page width (page-1 issue)
            wide_bright = fw > 0.50 and mean_gray > 225
            sparse_text = dark_pct < 18 and box_area(r['bbox']) > min_abs_area * 0.5
            if wide_bright or sparse_text:
                if args.verbose >= 1:
                    print(f"  FILTER (text blob): [{bw}x{bh}] {fw:.0%} width "
                          f"mean_gray={mean_gray:.0f} dark={dark_pct:.0f}% bright={bright_pct:.0f}% → text")
                continue
            # Also reject uniform-white blobs (no real graphical content)
            if np.std(arr, axis=(0, 1)).mean() <= 5:
                continue
        graphical.append(r)

    # Relaxed fallback if nothing found
    if not graphical:
        if args.verbose >= 1:
            print("  Trying relaxed filter (min-area-pct * 0.3)...")
        relaxed = min_abs_area * 0.3
        graphical = [r for r in merged
                     if box_area(r['bbox']) >= relaxed
                     and 0.15 < r['bbox'][2] / max(r['bbox'][3], 1) < 10
                     and r['bbox'][0] > w * 0.01
                     and r['bbox'][0] + r['bbox'][2] < w * 0.99
                     and r['bbox'][1] > h * 0.01
                     and r['bbox'][1] + r['bbox'][3] < h * 0.99
                     # Skip wide-rect text paragraphs in relaxed mode
                     and not (r['bbox'][2] / max(w, 1) > 0.70
                              and r['bbox'][0] / max(w, 1) < 0.10
                              or r['bbox'][2] / max(w, 1) > 0.70
                              and (w - r['bbox'][0] - r['bbox'][2]) / max(w, 1) < 0.10
                             )
                     and not (r['bbox'][2] / max(w, 1) > 0.70
                              and 1.0 < r['bbox'][2] / max(r['bbox'][3], 1) < 3.0)]

        # Remove regions contained in larger graphical regions
        final = []
        for r in graphical:
            dominated = False
            for other in graphical:
                if r is other:
                    continue
                if box_area(r['bbox']) >= box_area(other['bbox']):
                    continue
                if containment(other['bbox'], r['bbox']) > 0.85:
                    dominated = True
                    break
            if not dominated:
                final.append(r)
        graphical = final
        if args.verbose >= 1:
            print(f"  Relaxed filter: {len(graphical)} graphical objects")

    # Sort in reading order
    ordered = sort_reading_order(graphical) if graphical else []
    print(f"Detected {len(ordered)} graphical object(s) in reading order.")
    for i, r in enumerate(ordered):
        x, y, bw, bh = r['bbox']
        print(f"  {i+1}. [{x},{y},{bw}x{bh}]  area={bw*bh:,}  sources={r['sources']}")

    # Save cropped objects
    for i, r in enumerate(ordered):
        x, y, bw, bh = r['bbox']
        obj = img[y:y + bh, x:x + bw]
        outpath = os.path.join(out_dir, f'{base}_graph_{i+1:02d}.png')
        cv2.imwrite(outpath, obj)

    # text_blocks already detected earlier; reused here for overlap resolution.

    # ---- Resolve overlaps between text blocks and graphical objects ----
    tt_stats = {'overlaps_found': 0, 'overlaps_fixed': 0}
    tg_stats = {'overlaps_found': 0, 'overlaps_fixed': 0}

    if args.verbose >= 2:
        text_pre = [tuple(b['bbox']) for b in text_blocks]
    else:
        text_pre = None

    # Initialize text_mid before conditional block so it's always bound
    text_mid: list[tuple[int, ...]] = []

    if text_blocks and ordered:
        # Text-text overlaps
        text_resolved, tt_stats = resolve_text_text_overlaps(text_blocks)

        # Capture intermediate state after text-text resolution for verbose >= 2 attribution
        text_mid = [tuple(b['bbox']) for b in text_blocks]
        for i, tb in enumerate(text_resolved):
            text_blocks[i]['bbox'] = tuple(tb['bbox'])

        # Text-graph overlaps (graphical objects take precedence)
        text_blocks, tg_stats = resolve_text_graph_overlaps(text_blocks, ordered, page_w=w, page_h=h)

    total_found = tt_stats['overlaps_found'] + tg_stats['overlaps_found']
    total_fixed = tt_stats['overlaps_fixed'] + tg_stats['overlaps_fixed']
    if args.verbose >= 1 and text_blocks:
        print(f"Overlaps: {total_found} found, {total_fixed} fixed "
              f"(text-text: {tt_stats['overlaps_fixed']}, "
              f"text-graph: {tg_stats['overlaps_fixed']})")

    # Per-block overlap resolution details at verbose level >= 2
    if args.verbose >= 2 and text_pre and ordered:
        print(f"\nText block overlap resolution (per-block detail):")
        for i, (tb_pre, tb_post) in enumerate(zip(text_pre, text_blocks)):
            px, py, pw, ph = text_pre[i]
            nx, ny, nw, nh = tuple(tb_post['bbox'])

            # Determine remaining overlaps after resolution
            rem_graphs = []
            for gi, g in enumerate(ordered):
                gx, gy, gw, gh = g['bbox']
                xi = max(nx, gx); yi = max(ny, gy)
                xe = min(nx + nw, gx + gw); ye = min(ny + nh, gy + gh)
                if (xe > xi) and (ye > yi):
                    rem_graphs.append(f'graph[{gi+1}]({(xe-xi)*(ye-yi)}px)')

            # Compute what changed by comparing pre/mid/post phases separately
            w_changed = (pw != nw)
            h_changed = (ph != nh)
            # x_mid, y_mid, w_mid, h_mid are populated below inside the `if text_mid:` guard
            x_mid = y_mid = w_mid = h_mid = px  # default to pre-values

            # Determine which phase caused what change
            causes = []
            if text_mid:
                # Phase 1: text-text changes
                mid_x_changed = (x_mid != px)
                mid_y_changed = (y_mid != py)
                mid_w_changed = (w_mid != pw)
                mid_h_changed = (h_mid != ph)
                if mid_w_changed or mid_h_changed:
                    causes.append(f'text-text dim-change')
                    # Find original text overlap partners
                    for j in range(len(text_pre)):
                        if i == j: continue
                        xj, yj, wj, hj = text_pre[j]
                        xi2 = max(px, xj); yi2 = max(py, yj)
                        xe2 = min(px + pw, xj + wj); ye2 = min(py + ph, yj + hj)
                        if (xe2 > xi2) and (ye2 > yi2):
                            causes.append(f'text#{j+1}')

                # Phase 2: graph changes = post minus mid state
                gx2, gy2, gw2, gh2 = text_mid[i]

                # Check if ANY mid→post component changed (w/h AND/OR x/y)
                dim_changed = (gw2 != nw or gh2 != nh)
                pos_changed = (gx2 != nx or gy2 != ny)
                if dim_changed or pos_changed:
                    for gi, g in enumerate(ordered):
                        grx, gry, grw, grh = g['bbox']
                        # Compare mid-bbox with graph to determine shrink direction
                        xix = max(gx2, grx); yiy = max(gy2, gry)
                        xex = min(gx2 + gw2, grx + grw); yey = min(gy2 + gh2, gry + grh)
                        ix_val = max(0, xex - xix); iy_val = max(0, yey - yiy)
                        if ix_val == 0 or iy_val == 0: continue
                        # Mimic resolve_text_graph_overlaps logic: shorter dim gets shrunk
                        if ix_val <= iy_val:
                            side = 'LEFT' if (xix <= grx + grw / 2) else 'RIGHT'
                            causes.append(f'graph[{gi+1}] {side}({ix_val}x{iy_val})')
                        else:
                            side = 'TOP' if (yiy <= gry + grh / 2) else 'BOTTOM'
                            causes.append(f'graph[{gi+1}] {side}({ix_val}x{iy_val})')

                    # If position changed but dimensions didn't, it was a fallback move
                    # (_try_move_vertically or _try_move_horizontally after sub-threshold shrink)
                    if pos_changed and not dim_changed:
                        for gi, g in enumerate(ordered):
                            grx, gry, grw, grh = g['bbox']
                            xix = max(gx2, grx); yiy = max(gy2, gry)
                            xex = min(gx2 + gw2, grx + grw); yey = min(gy2 + gh2, gry + grh)
                            ix_val = max(0, xex - xix); iy_val = max(0, yey - yiy)
                            if ix_val == 0 or iy_val == 0: continue
                            if iy_val > ix_val:
                                ny_below = gry + grh + 4
                                ny_above = max(0, gry - gh2 - 4)
                                moved_down = (gy2 + gh2 <= grx + grw) or \
                                    (ny >= ny_below) or (ny == 0 and abs(ny - gy2) > abs(ny_above - gy2))
                                causes.append(f'graph[{gi+1}] move {"down" if moved_down else "up"} '
                                              f'(shrunk-w → {gw2}x{nw})')
                            else:
                                nx_right = grx + grw + 4
                                nx_left = max(0, grx - gw2 - 4)
                                moved_right = (gx2 + gw2 <= grx) or \
                                    (nx >= nx_right) or (nx == 0 and abs(nx - gx2) > abs(nx_left - gx2))
                                causes.append(f'graph[{gi+1}] move {"right" if moved_right else "left"} '
                                              f'(shrunk-h → {gh2}x{nh})')

            if not causes:
                action = 'unchanged'
            elif nw == 0 or nh == 0:
                graph_names = [f'graph[{gi+1}]' for gi, _ in enumerate(ordered)]
                action = f'HIDDEN by graph ({", ".join(graph_names)})'
            else:
                action = f'({"; ".join(causes)})'

            print(f"  [{i+1}] [{px},{py},{pw}x{ph}] -> [{nx},{ny},{nw}x{nh}] ({action})")

    overlaps_summary = {
        'text_text': {
            'found': tt_stats['overlaps_found'],
            'fixed': tt_stats['overlaps_fixed'],
        },
        'text_graph': {
            'found': tg_stats['overlaps_found'],
            'fixed': tg_stats['overlaps_fixed'],
        },
    }

    # ---- Annotation: text regions (after overlap resolution) ----
    if text_blocks:
        draw_overlay(img, text_blocks, 'text',
                     os.path.join(out_dir, f'{base}_text_regions.png'))

    # ---- Annotation ----
    if ordered:
        draw_overlay(img, ordered, 'graph',
                     os.path.join(out_dir, f'{base}_annotated.png'))

    # ---- Debug masks ----
    if args.debug:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imwrite(os.path.join(out_dir, f'{base}_debug_color.png'),
                    (hsv[:, :, 1] > args.saturation_threshold).astype(np.uint8) * 255)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.0), 50, 200)
        cv2.imwrite(os.path.join(out_dir, f'{base}_debug_edges.png'),
                    cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2))
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 10)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        cv2.imwrite(os.path.join(out_dir, f'{base}_debug_cc.png'),
                    cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=4))

    # ---- Summary JSON ----
    summary = {
        'image': args.image,
        'image_size': {'width': w, 'height': h},
        'min_area_pct': args.min_area_pct,
        'num_graphical_objects': len(ordered),
        'graphical_objects': [
            {'index': i+1, 'bbox': {'x': r['bbox'][0], 'y': r['bbox'][1],
                                     'width': r['bbox'][2], 'height': r['bbox'][3]},
             'area': box_area(r['bbox']), 'sources': r['sources'],
             'output_file': f'{base}_graph_{i+1:02d}.png'}
            for i, r in enumerate(ordered)],
        'num_text_blocks': len(text_blocks),
        'text_blocks': [{'bbox': {'x': b['bbox'][0], 'y': b['bbox'][1],
                                  'width': b['bbox'][2], 'height': b['bbox'][3]}}
                        for b in text_blocks],
        'overlap_resolution': overlaps_summary,
    }
    summary_path = os.path.join(out_dir, f'{base}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- Assemble layout-preserving HTML (text + images at original positions) ----
    ocr_backend = f"Ollama ({args.ollama_model})" if args.ollama_model else "Tesseract"
    if args.verbose >= 1:
        print(f"OCR backend: {ocr_backend}")
    html_output = layout_html.assemble_layout_html(
        img, h, w, ordered, text_blocks,
        model=args.ollama_model, tess_lang=args.tess_lang,
        use_system_tess=args.tess_system,
    )
    html_path = os.path.join(out_dir, f'{base}_layout.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_output)
    if args.verbose >= 1:
        print(f"Saved layout HTML to {html_path}")


if __name__ == '__main__':
    main()
