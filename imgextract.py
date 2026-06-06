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
        if y < h * 0.02 or (y + bh) > h * 0.98:
            continue
        if x < w * 0.02 or (x + bw) > w * 0.98:
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
    parser.add_argument('--verbose', action='store_true', help='Verbose debug output')
    parser.add_argument('--debug', action='store_true', help='Save intermediate debug masks')
    args = parser.parse_args()

    img = load_image(args.image)
    h, w = img.shape[:2]
    base = image_base(args.image)
    out_dir = args.output or os.path.dirname(os.path.abspath(args.image))
    os.makedirs(out_dir, exist_ok=True)

    min_abs_area = h * w * args.min_area_pct
    page_frac_threshold = 0.50  # reject any bbox covering >50% of the page
    max_object_frac = 0.25  # max area fraction for a single graphical object

    if args.verbose:
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
        if args.verbose:
            print(f"  {name:8s} regions: {len(result)}")
        regions.extend(result)

    if args.verbose:
        print(f"  Total raw regions: {len(regions)}")

    # Pre-filter: remove page-spanning regions BEFORE merging
    total_area = h * w
    pre_merged = []
    for r in regions:
        area = box_area(r['bbox'])
        if area > total_area * page_frac_threshold:
            if args.verbose:
                print(f"  PRE-FILTER (page-spanning): [{r['bbox'][2]}x{r['bbox'][3]}] {area/total_area:.0%}")
            continue
        if area < min_abs_area:
            if args.verbose:
                print(f"  PRE-FILTER (area): [{r['bbox'][2]}x{r['bbox'][3]}] {area} < {min_abs_area:.0f}")
            continue
        pre_merged.append(r)

    if args.verbose:
        print(f"  After pre-filter: {len(pre_merged)} / {len(regions)} regions")

    # Merge nearby regions
    merged = merge_regions(pre_merged, gap_thresh=args.gap_threshold)

    if args.verbose:
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

    if args.verbose:
        print(f"  After dedup: {len(post_merge)} regions")

    # Filter to final graphical objects
    graphical = []
    for r in post_merge:
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
        roi = img[y:y + bh, x:x + bw]
        if roi.size > 0 and np.std(roi.astype(np.float64), axis=(0, 1)).mean() <= 5:
            continue
        graphical.append(r)

    # Relaxed fallback if nothing found
    if not graphical:
        if args.verbose:
            print("  Trying relaxed filter (min-area-pct * 0.3)...")
        relaxed = min_abs_area * 0.3
        graphical = [r for r in merged
                     if box_area(r['bbox']) >= relaxed
                     and 0.15 < r['bbox'][2] / max(r['bbox'][3], 1) < 10
                     and r['bbox'][0] > w * 0.01
                     and r['bbox'][0] + r['bbox'][2] < w * 0.99
                     and r['bbox'][1] > h * 0.01
                     and r['bbox'][1] + r['bbox'][3] < h * 0.99]
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
        if args.verbose:
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

    # ---- Detect text blocks ----
    text_blocks = detect_text_blocks(img, min_area=args.min_text_area)
    if args.verbose:
        print(f"Detected {len(text_blocks)} text block(s).")
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
    }
    summary_path = os.path.join(out_dir, f'{base}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
