<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:003049,50:00798c,100:30c9a8&height=180&section=header&text=MCQ%20Paper%20Checker&fontSize=46&fontColor=ffffff&fontAlignY=38&desc=OpenCV-Based%20OMR%20Sheet%20Grading%20%7C%20Streamlit%20Web%20App&descAlignY=58&descSize=18&descColor=c2f2e8" width="100%"/>

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

**A computer-vision-based Optical Mark Recognition (OMR) grading system that takes a photo of a multiple-choice answer sheet and automatically detects, grades, and scores it — accessible through an interactive Streamlit web app.**

</div>

---

## 🖼️ Sample Sheet

> The blank answer sheet template the system is designed to recognize and grade.

<div align="center">
<img src="https://raw.githubusercontent.com/Rhythm05Roy/Mcq-Paper-Checker/main/MCQPaper-Photoroom.png" width="55%" alt="Blank MCQ answer sheet template"/>
</div>

A 5-question × 5-choice (A–E) bubble grid, a `Name` field, and a `GRADE` box that the system overlays with the computed score after grading.

---

## Overview

MCQ Paper Checker is a classic image-processing pipeline built on **OpenCV**, wrapped in a **Streamlit** web interface. Given a photo of a filled-in answer sheet, it:

1. Detects the sheet and the grade box as the two largest rectangular contours in the image
2. Applies a perspective transform to flatten and align the answer-bubble grid
3. Thresholds the warped image to isolate pencil/pen marks
4. Splits the grid into individual answer cells and counts filled (non-zero) pixels per cell
5. Picks the most-marked option per question as the student's answer
6. Compares against a configurable answer key and computes a percentage score

The repository includes both **web app** implementations (`app.py`, `app2.py`) and a **live webcam/standalone OpenCV script** (`omr.py`) that visualizes every stage of the pipeline side-by-side and overlays the grade directly onto the original image.

---

## Pipeline Architecture

```
                    Uploaded / Captured Image
                              │
                              ▼
                  ┌────────────────────────┐
                  │  Resize (500×500)       │
                  │  Grayscale → Blur →     │
                  │  Canny Edge Detection   │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │  Find Contours →         │
                  │  rectContour()           │  ── Filters by area & 4-corner shape,
                  │                          │     sorts by size (descending)
                  └────────────┬────────────┘
                               │
              ┌────────────────┴─────────────────┐
              ▼                                   ▼
   Largest rectangle                    2nd-largest rectangle
   = Answer Grid                         = Grade Box
              │                                   │
              ▼                                   ▼
   getCornerPoints() + reorder()        getCornerPoints() + reorder()
              │                                   │
              ▼                                   ▼
   Perspective Warp → 500×500           Perspective Warp → 500×500
   Grayscale → Threshold (inv, 190)
              │
              ▼
   splitBoxes() → 5×5 grid of cells
              │
              ▼
   countNonZero() per cell → myPixelVal
              │
              ▼
   argmax per row → marked answer per question
              │
              ▼
   Compare vs. answer key → grading[] → score (%)
              │
              ▼
   Display results (Streamlit) /
   Overlay score onto Grade Box (omr.py)
```

---

## Project Structure

```text
Mcq-Paper-Checker/
├── app.py                  # Minimal Streamlit app — single-rectangle detection, fixed 5×5 key
├── app2.py                 # Full Streamlit app — sidebar config, dynamic answer key, results table
├── omr.py                  # Standalone/webcam OpenCV script — visualizes every pipeline stage
├── utils.py                # Shared image-processing helpers (contours, warping, splitting)
├── tempCodeRunnerFile.py    # Leftover debug snippet from development
├── MCQPaper-Photoroom.png    # Blank answer sheet template (background-removed)
├── 1.jpg, 2.jpg, 3.jpg        # Sample filled-in answer sheet photos
├── 2-Photoroom.png, 3-Photoroom.png   # Background-removed sample sheets
├── FinalImage.jpg                      # Example graded output (score overlaid on sheet)
└── README.md
```

---

## Component Details

### `utils.py` — Shared Image-Processing Helpers

| Function | Purpose |
|---|---|
| `rectContour(contours)` | Filters detected contours to those with area > 50 and exactly 4 corners (i.e. rectangles), sorted largest-first |
| `getCornerPoints(contour)` | Approximates a contour's polygon and returns its corner points |
| `reorder(points)` | Reorders 4 corner points into a consistent `[top-left, top-right, bottom-left, bottom-right]` order based on coordinate sums/differences — required for a correct perspective transform |
| `splitBoxes(img)` | Splits a warped, thresholded grid image into a 5×5 array of individual answer-bubble cells |
| `showAnswers(img, myIndex, grading, ans, questions, choices)` | Draws colored circles on the original image — green for correct, red for incorrect (with the correct answer also marked in green) |
| `stackImages(imgArray, scale, labels)` | Combines a grid of intermediate-stage images (with labels) into a single debug visualization — used by `omr.py` |

### `app.py` — Minimal Web App

The simplest implementation: detects a **single** largest rectangle (the answer grid only), applies a fixed perspective warp, thresholds, splits into a 5×5 grid, and grades against a **hardcoded answer key** (`[1, 2, 0, 1, 4]`, i.e. B, C, A, B, E). Displays only the final percentage score.

### `app2.py` — Full-Featured Web App

The more complete implementation, recommended for general use:

- **Sidebar configuration** — set the number of questions (1–20) and choices per question (2–10)
- **Interactive answer key** — select the correct option for each question via dropdowns
- **Two-rectangle detection** — locates both the **answer grid** (largest contour) and the **grade box** (second-largest contour), each independently reordered and warped
- **Detailed results table** — shows marked vs. correct answer per question with ✅/❌ status, plus summary metrics for score and questions-correct

### `omr.py` — Standalone Visualization / Webcam Script

A non-Streamlit OpenCV script intended for live webcam input (or a static image via `path = '1.jpg'`). It:

- Runs the full pipeline continuously in a loop
- Uses `stackImages()` to display a 3×4 grid of every intermediate stage (original, grayscale, blur, Canny, contours, biggest contour, warped grid, threshold, result, raw drawing, inverse warp, final)
- Draws the computed score directly onto the grade box region via an **inverse perspective warp**, blending it back onto the original image (`imgFinal`)
- Saves the final annotated frame to `FinalImage.jpg` when the **`s`** key is pressed, and exits on **`q`**

> `omr.py` is configured for a specific webcam device (`cv2.VideoCapture('/dev/video2')`) and will need to be adapted to your camera index/path, or pointed at a static image by setting `webCamFeed = False`.

---

## Getting Started

### Prerequisites

- Python 3.10+
- A webcam (only required for `omr.py`'s live mode)

### Installation

```bash
git clone https://github.com/Rhythm05Roy/Mcq-Paper-Checker.git
cd Mcq-Paper-Checker

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install streamlit opencv-python numpy
```

> **Note:** This repository does not currently include a `requirements.txt`. The runtime dependencies (verified against the project's environment) are `streamlit`, `opencv-python`, and `numpy`.

### Running the Web App

```bash
# Minimal version (fixed 5x5 grid, hardcoded answer key)
streamlit run app.py

# Full version (configurable grid size, dynamic answer key, results table)
streamlit run app2.py
```

### Running the Standalone/Webcam Script

```bash
python omr.py
```

Edit the top of `omr.py` to match your setup:

```python
path = '1.jpg'              # used if webCamFeed = False
questions, choices = 5, 5
ans = [1, 2, 0, 1, 4]        # 0-indexed correct answers (A=0, B=1, C=2, D=3, E=4)
webCamFeed = True             # set False to grade a static image instead
cap = cv2.VideoCapture('/dev/video2')   # change to your webcam index, e.g. 0
```

---

## How It Works — Step by Step

1. **Upload / capture** an image of a filled-in answer sheet (see the sample template above)
2. **Preprocess** — resize to 500×500, convert to grayscale, apply Gaussian blur, run Canny edge detection
3. **Detect rectangles** — find external contours, filter for 4-corner shapes above a minimum area, sort by size
4. **Warp perspective** — reorder each rectangle's corners and warp it to a flat 500×500 image (one warp for the answer grid, one for the grade box in `app2.py`/`omr.py`)
5. **Threshold** — convert the warped grid to grayscale and apply an inverted binary threshold (cutoff `190`) so filled bubbles become white pixels
6. **Split into cells** — divide the thresholded grid into a `questions × choices` array of cells via `np.vsplit`/`np.hsplit`
7. **Count marks** — for each cell, count non-zero pixels; the cell with the most marked pixels per row is the selected answer
8. **Grade** — compare selected answers against the answer key, compute `correct / total × 100`
9. **Display** — show the score (and, in `app2.py`, a per-question breakdown) in the Streamlit UI, or overlay it onto the sheet image in `omr.py`

---

## Example Output

<div align="center">
<table>
<tr>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/Rhythm05Roy/Mcq-Paper-Checker/main/1.jpg" width="100%" alt="Sample filled answer sheet"/>
<br/><b>Input</b> — Photographed answer sheet
</td>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/Rhythm05Roy/Mcq-Paper-Checker/main/FinalImage.jpg" width="100%" alt="Graded output with score overlay"/>
<br/><b>Output</b> — Graded, with score overlaid on the sheet (<code>omr.py</code>)
</td>
</tr>
</table>
</div>

In the graded output, correctly answered questions are circled in **green**; incorrect ones are circled in **red**, with the correct option also highlighted in green for reference.

---

## Default Answer Key

`app.py` and `omr.py` use the following hardcoded 5-question key (0-indexed: A=0, B=1, C=2, D=3, E=4):

| Question | Correct Option |
|---|---|
| 1 | B (1) |
| 2 | C (2) |
| 3 | A (0) |
| 4 | B (1) |
| 5 | E (4) |

`app2.py` instead lets you set this key interactively via the sidebar for any grid size from 1–20 questions and 2–10 choices.

---

## Known Limitations & Future Enhancements

- The pipeline assumes a clean, high-contrast photo with the sheet's edges fully visible and the largest contours corresponding to the answer grid (and grade box)
- `app.py` only locates one rectangle and uses a fixed 5×5 layout and hardcoded key
- `omr.py` is hardcoded to a specific webcam device path and a fixed answer key
- No `requirements.txt` is currently provided
- Planned improvements: configurable answer key formats, more robust error handling, and support for sheets with varying layouts and grid sizes across both app variants

---

## License

MIT — see `LICENSE`.

## Acknowledgments

- **OpenCV** for the image processing and contour-detection pipeline
- **Streamlit** for the interactive web interface
- Answer sheet template courtesy of *Murtaza's Workshop*

<div align="center">

**Built by [Ridam Roy](https://github.com/Rhythm05Roy)**

[![GitHub](https://img.shields.io/badge/GitHub-Rhythm05Roy-181717?style=flat-square&logo=github)](https://github.com/Rhythm05Roy)
[![Email](https://img.shields.io/badge/Email-ridam15--4260%40diu.edu.bd-D44638?style=flat-square&logo=gmail&logoColor=white)](mailto:ridam15-4260@diu.edu.bd)

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:003049,50:00798c,100:30c9a8&height=80&section=footer" width="100%"/>

</div>
