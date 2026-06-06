# AIConvert - PDF to Standard Textbook Pages

Convert scanned textbook PDFs into standard PDF pages that preserve the original layout while separating graphical objects from text content.

## Pipeline

| Step | File | Description |
|------|---|------|
| 1 | `pdf2png.sh` | Convert input PDF into one PNG image per page |
| 2 | `imgextract.py` | Detect and crop graphical objects from each page |
| 3 | `textextract.sh` | Extract text from each page using Tesseract |
| 4 | `textcleanup.py` | Correct extracted text using Ollama / LLM |
| 5 | `buildpage.sh` | Combine cropped images + corrected text into a standard PDF page |
| 6 | `assemblepdf.sh` | Stitch all pages into a single output PDF |

## imgextract.py - Graphical Object Detection

Detects graphical objects (color-coded tables, diagrams, charts, figures) in page images and crops them out in reading order (top-to-bottom, left-to-right within each row).

### Detection Pipeline

Four complementary OpenCV-based strategies detect graphical objects, then results are merged and filtered:

| Strategy | What it finds | How it works |
|--|---|--|
| **Color saturation** | Color-coded tables, highlighted regions | HSV saturation thresholding + morphological closing |
| **Edge detection** | Bordered diagrams, line art | Canny edges + contour bounding boxes |
| **Connected components** | White/bright region groups | Adaptive thresholding + morphological closing |
| **Gradient analysis** | Diagrams with directional structure | Sobel gradient magnitude thresholding |

### Post-Processing

1. **Pre-filter** - removes page-spanning regions (arrow borders, header/footer noise)
2. **Merge** - combines nearby/overlapping regions (with containment guard)
3. **IoU dedup** - removes regions mostly contained within a larger one
4. **Final filter** - area, aspect ratio, margin, and color variation constraints
5. **Sort** - reading order (top-to-bottom, left-to-right)
6. **Text block detection** - separately identifies preserved text regions

### Usage

```bash
# Basic usage
python3 imgextract.py page-1.png -o output_dir/

# With verbose output
python3 imgextract.py page-1.png -o output_dir/ --verbose

# Save intermediate debug masks
python3 imgextract.py page-1.png -o output_dir/ --debug

# Tunable parameters
python3 imgextract.py page-1.png \
  -o output_dir/ \
  --min-area-pct 0.005 \
  --gap-threshold 0.12 \
  --saturation-threshold 30
```

### Parameters

| Parameter | Default | Description |
|------|------|------|
| **min-area-pct** | 0.005 | Minimum area fraction of page for a graphical object |
| **gap-threshold** | 0.12 | Proximity gap for merging nearby regions (0-1) |
| **saturation-threshold** | 30 | HSV saturation threshold for color detection |
| **verbose** | off | Print detailed detection logs |
| **debug** | off | Save intermediate color/edge/CC masks |

### Output Files (per page)

| File | Description |
|------|------|
| **pageN_graph_M.png** | Cropped graphical objects, numbered in reading order |
| **pageN_text_regions.png** | Bounding boxes around detected text blocks |
| **pageN_annotated.png** | Overlay showing all detected graphical regions |
| **pageN_summary.json** | Machine-readable layout analysis (JSON) |

### Test Results

Tested on sample pages (1651x1275):

| Page | Objects Found | Description |
|------|------|------|
| 1 | 1 | Color-coded node specification table (left column) |
| 2 | 2 | Color bar chart (left) + green "Performance" highlight |
| - | 20/17 | Text blocks detected for both pages |

### Requirements

- Python 3.8+
- OpenCV (`pip install opencv-python-headless`)
- NumPy
