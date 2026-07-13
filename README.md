# FaceVision — Real-Time Face Detection System

A real-time face, eye, and smile detection system built with Python and OpenCV. Supports two detection backends — a DNN-based ResNet SSD detector for higher accuracy and Haar Cascade classifiers as a lightweight fallback. Features a HUD overlay with live stats and switchable detectors.

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
Opens the webcam, detects faces frame by frame using either the DNN detector or Haar cascades (switchable at runtime with `D` key). Draws styled bounding boxes with confidence scores, and overlays a HUD. You can toggle eye/smile detection, flip the camera, and capture screenshots.

**Image Mode** (`detect_image.py`):
Takes a single image or a folder of images, runs face + eye detection, saves annotated copies. Automatically uses DNN if the model is available.

Two detection backends:

1. **DNN (ResNet SSD)** — Deep learning-based. Higher accuracy, handles angles and lighting variation well. Shows confidence percentage per face. This is the default when the model is present.
2. **Haar Cascade** — Classic Viola-Jones approach. Faster but less accurate. Works as fallback if DNN model isn't downloaded.

For each detected face, eye and smile cascades run within the face ROI to find finer features.

---

## Technologies Used

| Tech | Purpose |
|------|---------|
| Python 3.10+ | Main language |
| OpenCV 4.5+ (with DNN module) | Image processing, Haar cascades, DNN inference |
| NumPy | Array operations |

No separate deep learning frameworks needed — OpenCV's built-in DNN module handles inference. Runs on CPU.

---

## Dataset / Pre-trained Models

No external training dataset is required. We use pre-trained models:

**DNN model** (downloaded separately, ~10MB):
- `res10_300x300_ssd_iter_140000.caffemodel` — ResNet-10 SSD face detector, trained on multiple face datasets
- `deploy.prototxt` — model architecture definition

**Haar Cascades** (bundled with OpenCV):
- `haarcascade_frontalface_default.xml` — face detection
- `haarcascade_eye.xml` — eye detection
- `haarcascade_smile.xml` — smile detection

---

## Methodology

### DNN Detector (ResNet-10 SSD)

The primary detector uses a Single Shot Multibox Detector (SSD) with a ResNet-10 backbone, loaded via `cv2.dnn.readNetFromCaffe`. The input image is resized to 300x300, converted to a blob with mean subtraction, and passed through the network in a single forward pass. Detections above the confidence threshold (default 0.5) are kept.

This approach handles varying lighting, partial occlusion, and tilted faces much better than Haar cascades.

### Haar Cascade (Viola-Jones — Fallback)

The fallback detector uses the Viola-Jones algorithm:

1. **Haar features** — Rectangular filters that capture contrast patterns (eyes darker than forehead, etc.).
2. **Integral image** — Enables fast feature computation at any scale.
3. **AdaBoost** — Selects the most discriminative features from ~180K candidates.
4. **Cascade** — Multi-stage classifier; 95% of non-face regions are rejected in the first two stages.

### Key Parameters

| Param | DNN | Haar |
|-------|-----|------|
| Confidence threshold | 0.5 | N/A |
| scaleFactor | N/A | 1.1 |
| minNeighbors | N/A | 5 |
| minSize | 300x300 (input) | 30x30 |

All parameters are in `config.py`.

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

# download the DNN model (~10MB, one-time)
python download_models.py
```

The DNN model download is optional — the app falls back to Haar cascades if the model isn't present.

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
| D | Switch between DNN and Haar detector |

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
- DNN detector handles lighting changes, angles, and partial occlusion well
- Real-time detection at 25-30 FPS with DNN on CPU
- Confidence scores shown per face (DNN mode)
- Multi-face detection in the same frame
- Eye and smile detection within face ROI
- Color-coded bounding boxes (green = close, cyan = medium, yellow = far)
- Live switching between DNN and Haar detectors
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
├── download_models.py    # one-time DNN model download script
├── requirements.txt      # dependencies
├── .gitignore
├── LICENSE
├── README.md
├── models/               # DNN model files
│   ├── deploy.prototxt
│   └── *.caffemodel      # downloaded via download_models.py
├── screenshots/          # saved screenshots (auto-created)
└── output/               # batch processing output (auto-created)
```

---

## Future Scope

- Upgrade to YOLOv8-face or MediaPipe for even higher accuracy
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
2. Liu, W. et al. (2016). "SSD: Single Shot MultiBox Detector." ECCV.
3. OpenCV DNN Face Detector — https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
4. OpenCV Cascade Classifier Tutorial — https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
5. OpenCV Haar Cascades — https://github.com/opencv/opencv/tree/master/data/haarcascades

---

Built for HackZen Open Challenge 2026

