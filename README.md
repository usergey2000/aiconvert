# AIConvert - PDF to Standard Textbook Pages

Convert scanned textbook PDFs into standard PDF pages that preserve the original layout while separating graphical objects from text content.

## Pipeline

| Step | File | Description |
|------|------|------|
| 1 | `pdf2png.sh` | Convert input PDF into one PNG image per page |
| 2 | `imgextract.py` | Detect and crop graphical objects from each page |
| 3 | `textextract.py` | Extract text from each page using Tesseract or Ollama vision model |
| 4 | `textcleanup.py` | Correct extracted text using Ollama / LLM |
| 5 | `buildpage.sh` | Combine cropped images + corrected text into a standard PDF page |
| 6 | `assemblepdf.sh` | Stitch all pages into a single output PDF |

## imgextract.py - Graphical Object Detection

Detects graphical objects (color-coded tables, diagrams, charts, figures) in page images and crops them out in reading order (top-to-bottom, left-to-right within each row).

### Detection Pipeline

Four complementary OpenCV-based strategies detect graphical objects, then results are merged and filtered:

| Strategy | What it finds | How it works |
|----------|---------------|--------------|
| **Color saturation** | Color-coded tables, highlighted regions | HSV saturation thresholding + morphological closing |
| **Edge detection** | Bordered diagrams, line art | Canny edges + contour bounding boxes |
| **Connected components** | White/bright region groups | Adaptive thresholding + morphological closing |
| **Gradient analysis** | Diagrams with directional structure | Sobel gradient magnitude thresholding |

### Post-Processing

1. **Pre-filter** - removes page-spanning regions (arrow borders, header/footer noise)
2. **Merge** - combines nearby/overlapping regions (with containment guard)
3. **Dedup** - removes regions mostly contained within a larger one
4. **Final filter** - area, aspect ratio, margin, and color variation constraints
5. **Sort** - reading order (top-to-bottom, left-to-right)
6. **Text block detection** - separately identifies text regions

### Multilingual Text Support

For pages with non-English text (e.g., Russian), use the `--tess-lang` option to specify the appropriate language code:

```bash
# Russian text
python3 imgextract.py page-1.png -o output_dir/ --tess-lang rus

# Multiple languages (comma-separated in Tesseract format)
python3 imgextract.py page-1.png -o output_dir/ --tess-lang eng+rus
```

Run `tesseract --list-langs` to see available language codes.

### Bug Fixes

- Removed dead code (unreachable `return` statement) in `_try_move_horizontally()` function

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

# OCR language (for text blocks in layout HTML)
python3 imgextract.py page-1.png -o output_dir/ --tess-lang rus

# Use system tesseract binary instead of Python library
python3 imgextract.py page-1.png -o output_dir/ --tess-system
```

### Parameters

| Parameter | Default | Description |
|-- ----|------|------|
| **min-area-pct** | 0.005 | Minimum area fraction of page for a graphical object |
| **gap-threshold** | 0.12 | Proximity gap for merging nearby regions (0-1) |
| **saturation-threshold** | 30 | HSV saturation threshold for color detection |
| **verbose** | off | Print detailed detection logs |
| **debug** | off | Save intermediate color/edge/CC masks |
| **tess-lang** | `eng` | Tesseract language code(s) for text OCR (e.g., `rus`, `deu+fra`) |
| **tess-system** | off | Use system tesseract binary directly instead of Python library |

### Overlap Resolution

The tool automatically detects and repairs text-block and text-graph overlaps:

- **Text-text**: later blocks are shifted to the minimum distance needed to escape their predecessors
- **Text-graph**: text blocks are shrunk on the shorter overlap dimension; if shrinking would produce a dimension < 50 px the block is moved along the longer axis
- **Page clamping**: any element pushed off-page is hidden (its content is already visible in the graph)

### Output Files (per page)

| File | Description |
|------|------|
| **pageN_graph_M.png** | Cropped graphical objects, numbered in reading order |
| **pageN_text_regions.png** | Bounding boxes around detected text blocks |
| **pageN_annotated.png** | Overlay showing all detected graphical regions |
| **pageN_summary.json** | Machine-readable layout analysis (JSON) |
| **pageN_layout.html** | Layout-preserving HTML with text and images at original positions |

### Test Results

Tested on sample pages (1651x1275):

| Page | Objects Found | Description |
|-- ----|------|------|
| 1 | 1 | Color-coded node specification table (left column) |
| 2 | 2 | Color bar chart (left) + green "Performance" highlight |
| - | 20/17 | Text blocks detected for both pages |

### Requirements

- Python 3.8+
- OpenCV (`pip install opencv-python-headless`)
- NumPy

## textextract.py - Vision-Model Text Extraction

Extract text from page images using an Ollama vision model (OCR + grammar correction + HTML formatting) in a single call.

### Vision Model Notes

**qwen3-vl models (2b, 4b, 8b)**: These standard vision language models are unstable for OCR tasks and may crash with "model runner has unexpectedly stopped" errors. This appears to be a known limitation with the qwen3-vl family when processing OCR workloads.

**Recommended model**: `maternion/LightOnOCR-2:1b` - A specialized OCR model that works reliably with this pipeline.

### Pipeline

| Stage | What it does |
|--------|--|
| **Deskew** | Hough-line angle detection; rotates image to correct page tilt |
| **Denoise** | Non-local means denoising (`fastNlMeansDenoising`) |
| **CLAHE** | Contrast-limited adaptive histogram equalization |
| **Downscale** | Largest dimension ≤ 1600 px — keeps the VL model's context window manageable |
| **Vision model** | Single `ollama.chat` call; model extracts all text, corrects OCR errors, and outputs a complete HTML document |

### Usage

```bash
# Basic usage (uses default model maternion/LightOnOCR-2:1b)
python3 textextract.py page-1.png

# Multiple images
python3 textextract.py page-1.png page-2.png -o output_dir/

# Custom Ollama model
python3 textextract.py page-1.png --model qwen3-vl:32b

# Short form
python3 textextract.py page-1.png -m qwen3-vl:32b

# Verbose output
python3 textextract.py page-1.png --verbose

# Dry run (shows what would be done)
python3 textextract.py page-1.png --dry-run
```

### Parameters

| Parameter | Default | Description |
|-- ----|------|------|
| **model, -m** | maternion/LightOnOCR-2:1b | Ollama vision model to use for OCR |
| **output, -o** | same dir as images | Output directory for HTML files |
| **verbose, -v** | off | Print detailed processing logs |
| **dry-run** | off | Show what would be done without running |

### Output Files (per image)

| File | Description |
|-- ----|------|
| **pageN_text.html** | Corrected, formatted HTML document |
| **pageN_preprocessed.png** | Deskewed / denoised / contrast-enhanced preview |

### Requirements

- Python 3.8+
- OpenCV, NumPy
- Ollama (running locally, model pulled with `ollama pull maternion/LightOnOCR-2:1b`)

## check_html.py - Layout Audit & Repair

Audit and repair layout problems in `imgextract.py` output HTML files.

### Problems Detected

| # | Problem | Example |
|---|---|---|
| 1 | Text-block ↔ text-block overlaps | Two adjacent paragraphs' bounding boxes intersect |
| 2 | Text-block ↔ graph-block overlaps | Text intruding into a graphical object's region |
| 3 | Zero / negative dimensions | A block shrunk below 1 px in either axis |
| 4 | Off-page placement | Element positioned left or above the page container |
| 5 | Off-page extent | Element extending right or bottom past the page boundary |

### Usage

```bash
# Check only (returns exit code 1 on problems)
python3 check_html.py page1_layout.html

# Check and fix (writes page1_layout-fixed.html)
python3 check_html.py page1_layout.html --fix

# Custom minimum dimension threshold
python3 check_html.py page1_layout.html --fix --min-dim 100
```

### Output

Without `--fix`: reports problems to stdout, exits 1.

With `--fix`: writes `<input_basename>-fixed.html` in the same directory (original file is never modified), prints the number of repairs made, and exits 0. Re-run without `--fix` to verify.

### Example

```bash
# Detect problems
$ python3 check_html.py imgextract-test/page1_layout.html
Found 1 problem(s):
  [Extends bottom past page edge: T17 (1298 > 1275)]  (467,1088,554x210)

# Fix problems
$ python3 check_html.py imgextract-test/page1_layout.html --fix
Fixed 1 issue(s). Output written to imgextract-test/page1_layout-fixed.html

# Verify clean
$ python3 check_html.py imgextract-test/page1_layout-fixed.html
OK: No layout problems found.
```

### Requirements

- Python 3.8+
