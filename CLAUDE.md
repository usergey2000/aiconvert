# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The task is to convert input pdf file (contains scans of textbook pages) into the standard pdf preserving format
of scanned pages.

## Project Overview

This repository contains :
1. pdf2png.sh      - convert input pdf file into a set of images, one image per page
2. textextract.sh  - to extract text from input image file using tesseract
3. imgextract.py   - to detect and crop graphical objects from page images (OpenCV-based)
4. textcleanup.py  - to correct extracted text
5. buildpage.sh    - to combine output from imgextact.sh and  textcleanup.py into standard pdf page
6. assemblepdf.sh  - to combine new pdf pages into a single document
4  textbook.pdf    - input textbook to work with