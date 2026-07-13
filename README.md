# FaceVision — Real-Time Face Detection System

A real-time face, eye, and smile detection system built with Python and OpenCV. Uses Haar Cascade classifiers to detect faces from webcam feed or static images, with a HUD overlay showing live stats.

Built for **HackZen Open Challenge 2026**.

---

## Team Details

| Role | Name |
|------|------|
| Developer | Abhinav E |
| Event | HackZen Open Challenge 2026 |
| Domain | Computer Vision |

---

## Problem Statement

Face detection is used everywhere — from phone unlock to surveillance systems and attendance tracking. But most existing implementations either need GPUs to run in real-time, or are too bare-bones to be useful out of the box.

We needed a solution that:
- Runs smoothly on a regular laptop without a GPU
- Goes beyond just drawing a box — detects eyes and smiles too
- Gives the user live stats and interactive controls
- Works on both live webcam feeds and saved images

---

## Objective

Build a lightweight, real-time face detection system that can:

1. Detect faces from webcam at 30+ FPS on CPU
2. Also detect eyes and smiles within face regions
3. Show a clean HUD with face count, FPS, and mode info
4. Let users take screenshots on the fly
5. Process static images and folders in batch mode

---

## Proposed Solution

FaceVision has two modes:

**Live Mode** (`face_detection.py`):
Opens the webcam, detects faces frame by frame, draws styled bounding boxes with corner accents, and overlays a HUD. You can toggle eye/smile detection, flip the camera, and capture screenshots — all with keyboard shortcuts.

**Image Mode** (`detect_image.py`):
Takes a single image or a folder of images, runs face + eye detection, saves annotated copies. Good for batch processing.

The detection pipeline is straightforward:

```
Camera/Image -> Grayscale -> Histogram Equalization -> Haar Cascade Detection -> Draw Annotations -> Display/Save
```

For faces detected, we also run eye and smile cascades within the face bounding box (ROI) to find finer features.

---

## Technologies Used

| Tech | Purpose |
|------|---------|
| Python 3.10+ | Main language |
| OpenCV 4.5+ | Image processing + Haar cascade detection |
| NumPy | Array operations |

No deep learning frameworks, no GPU needed.

---

## Dataset

No external dataset is required. We use the **pre-trained Haar Cascade XML files** that ship with OpenCV:

- `haarcascade_frontalface_default.xml` — face detection
- `haarcascade_eye.xml` — eye detection
- `haarcascade_smile.xml` — smile detection

These were trained using the Viola-Jones framework on large face/non-face image sets.

---

## Methodology

### Viola-Jones Framework (How Haar Cascades Work)

The detection is based on the Viola-Jones algorithm, which works in 4 steps:

1. **Haar features** — Rectangular filters that pick up contrast patterns (like eyes being darker than the forehead). There are edge, line, and four-rectangle variants.

2. **Integral image** — A fast way to compute the sum of any rectangular region in constant time. This makes evaluating thousands of Haar features practical.

3. **AdaBoost** — From ~180,000 possible features, AdaBoost picks the most useful ones and combines them into a strong classifier.

4. **Cascade structure** — Multiple stages of classifiers run in sequence. Easy negatives (clearly not a face) get rejected early, so the detector spends more time only on promising regions. About 95% of the image gets rejected in the first two stages.

### Detection Parameters

| Param | Value | What it does |
|-------|-------|-------------|
| scaleFactor | 1.1 | Shrinks image by 10% each pyramid level |
| minNeighbors | 5 | How many overlapping detections needed to confirm a face |
| minSize | (30,30) | Ignore anything smaller than 30x30 pixels |

All parameters are configurable in `config.py`.

---

## Installation & Setup

### Requirements
- Python 3.10+
- Webcam (for live mode)

### Steps

```bash
# clone the repo
git clone https://github.com/YOUR_USERNAME/FaceVision.git
cd FaceVision

# (optional) virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Linux/Mac

# install dependencies
pip install -r requirements.txt
```

That's it. Two packages, no GPU.

---

## Usage

### Live webcam detection

```bash
python face_detection.py
```

Controls:
| Key | Action |
|-----|--------|
| Q | Quit |
| S | Take screenshot |
| E | Toggle eye detection |
| M | Toggle smile detection |
| F | Flip camera (mirror) |

### Static image detection

```bash
# single image
python detect_image.py -i photo.jpg

# with custom output path
python detect_image.py -i photo.jpg -o result.jpg

# process entire folder
python detect_image.py -i path/to/images/

# headless (no display window)
python detect_image.py -i photo.jpg --no-display
```

### Run benchmark

```bash
python benchmark.py
```

Outputs detection speed at different resolutions + webcam FPS test. Saves results to `benchmark_report.json`.

---

## Results

### Detection Speed (measured on laptop CPU)

| Resolution | Avg Time | FPS |
|-----------|----------|-----|
| 320x240 | ~5ms | ~200 |
| 640x480 | ~15ms | ~65 |
| 1280x720 | ~40ms | ~25 |
| 1920x1080 | ~85ms | ~12 |

### What works well
- Real-time detection at 30+ FPS on 640x480
- Multi-face detection in the same frame
- Eye detection within face ROI
- Smile detection with visual label
- Color-coded bounding boxes (green = close, cyan = medium, yellow = far)
- Screenshot capture with flash effect
- Batch image processing

---

## Project Structure

```
FaceVision/
├── config.py             # all configurable parameters
├── face_detection.py     # main webcam detection app
├── detect_image.py       # static image / batch detection
├── benchmark.py          # performance benchmarking
├── requirements.txt      # dependencies
├── .gitignore
├── LICENSE
├── README.md
├── screenshots/          # saved screenshots (auto-created)
└── output/               # batch processing output (auto-created)
```

---

## Future Scope

- Replace Haar cascades with a DNN-based detector (SSD/YOLO) for better accuracy
- Add face recognition to identify known individuals
- Integrate emotion classification (happy, sad, angry etc.)
- Add age/gender estimation
- Build a web UI with Streamlit or Flask
- Face mask detection for safety compliance
- Multi-camera support
- Deploy as a REST API

---

## References

1. Viola, P., & Jones, M. (2001). "Rapid Object Detection using a Boosted Cascade of Simple Features." IEEE CVPR.
2. OpenCV Cascade Classifier Tutorial — https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
3. OpenCV Haar Cascades — https://github.com/opencv/opencv/tree/master/data/haarcascades

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Built for HackZen Open Challenge 2026
