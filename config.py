"""
Central configuration for FaceVision.
All tunable parameters are defined here.
"""

import cv2
import os

# App info
APP_NAME = "FaceVision"
APP_TITLE = "FaceVision - Real-Time Face Detection"
APP_EVENT = "HackZen Open Challenge 2026"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
BENCHMARK_REPORT_PATH = os.path.join(BASE_DIR, "benchmark_report.json")

# Haar cascade model paths (bundled with OpenCV)
HAAR_DIR = cv2.data.haarcascades
FACE_CASCADE_PATH = os.path.join(HAAR_DIR, "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH = os.path.join(HAAR_DIR, "haarcascade_eye.xml")
SMILE_CASCADE_PATH = os.path.join(HAAR_DIR, "haarcascade_smile.xml")

# DNN face detector (ResNet SSD — much more accurate than Haar)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DNN_PROTOTXT = os.path.join(MODELS_DIR, "deploy.prototxt")
DNN_CAFFEMODEL = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
DNN_CONFIDENCE_THRESHOLD = 0.5
DNN_INPUT_SIZE = (300, 300)
DNN_MEAN_VALUES = (104.0, 177.0, 123.0)

# set True to use DNN detector, False to fall back to Haar
USE_DNN = os.path.isfile(DNN_CAFFEMODEL)

# Webcam
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Face detection params
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (30, 30)

# Eye detection params
EYE_SCALE_FACTOR = 1.1
EYE_MIN_NEIGHBORS = 10
EYE_MIN_SIZE = (20, 20)
EYE_MAX_COUNT = 2

# Smile detection params
SMILE_SCALE_FACTOR = 1.7
SMILE_MIN_NEIGHBORS = 25
SMILE_MIN_SIZE = (25, 25)

# Face proximity thresholds (based on bounding box area)
FACE_AREA_CLOSE = 40000
FACE_AREA_MEDIUM = 15000

# Colors in BGR
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 200, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_OVERLAY = (40, 40, 40)
COLOR_ACCENT = (255, 140, 0)

COLOR_CLOSE = COLOR_GREEN
COLOR_MEDIUM = COLOR_CYAN
COLOR_FAR = COLOR_YELLOW

# Font and text
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_TITLE = 0.8
FONT_SCALE_NORMAL = 0.6
FONT_SCALE_LABEL = 0.5
FONT_SCALE_SMALL = 0.4
FONT_THICKNESS = 1
FONT_THICKNESS_BOLD = 2
LINE_TYPE = cv2.LINE_AA

# HUD layout
HUD_TOP_BAR_HEIGHT = 50
HUD_BOTTOM_BAR_HEIGHT = 35
HUD_OVERLAY_ALPHA = 0.6

HUD_TITLE_X = 10
HUD_TITLE_Y = 32
HUD_FACECOUNT_X = 200
HUD_FPS_X = 350
HUD_MODE_X = 470
HUD_CONTROLS_Y_OFFSET = 12
RECORDING_INDICATOR_RADIUS = 8
RECORDING_INDICATOR_X_OFFSET = 20

# Drawing
CORNER_ACCENT_LENGTH = 20
RECT_THICKNESS = 2
CORNER_THICKNESS = 2
CORNER_THICKNESS_IMAGE = 3

LABEL_PADDING_X = 5
LABEL_PADDING_Y = 5
LABEL_BG_PADDING = 10

# Screenshots
SCREENSHOT_PREFIX = "face_capture"
SCREENSHOT_FORMAT = ".png"
SCREENSHOT_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"
SCREENSHOT_FLASH_FRAMES = 10

# Image mode
IMAGE_WINDOW_NAME = "FaceVision - Image Detection"
IMAGE_MAX_DISPLAY_DIM = 800
IMAGE_SUMMARY_BAR_HEIGHT = 40
IMAGE_SUMMARY_Y_OFFSET = 12
SUPPORTED_IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

# Benchmark
BENCHMARK_RESOLUTIONS = [(320, 240), (640, 480), (1280, 720), (1920, 1080)]
BENCHMARK_ITERATIONS = 10
BENCHMARK_WEBCAM_DURATION = 5

# Defaults
DEFAULT_DETECT_EYES = True
DEFAULT_DETECT_SMILES = False
DEFAULT_FLIP_CAMERA = True

# Key binding for toggling detector
KEY_TOGGLE_DNN = ord('d')

# Key bindings
KEY_QUIT = ord('q')
KEY_SCREENSHOT = ord('s')
KEY_TOGGLE_EYES = ord('e')
KEY_TOGGLE_SMILE = ord('m')
KEY_FLIP_CAMERA = ord('f')

# UI strings
CONTROLS_TEXT = "[Q] Quit [S] Screenshot [E] Eyes [M] Smile [F] Flip [D] DNN/Haar"
SMILE_LABEL_TEXT = "Smiling :)"
SMILE_LABEL_Y_OFFSET = 25
