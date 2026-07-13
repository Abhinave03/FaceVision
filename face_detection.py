import cv2
import os
import time
import datetime
import numpy as np

from config import (
    APP_NAME, APP_TITLE, APP_EVENT,
    SCREENSHOT_DIR, FACE_CASCADE_PATH, EYE_CASCADE_PATH, SMILE_CASCADE_PATH,
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS, FACE_MIN_SIZE,
    EYE_SCALE_FACTOR, EYE_MIN_NEIGHBORS, EYE_MIN_SIZE, EYE_MAX_COUNT,
    SMILE_SCALE_FACTOR, SMILE_MIN_NEIGHBORS, SMILE_MIN_SIZE,
    FACE_AREA_CLOSE, FACE_AREA_MEDIUM,
    COLOR_GREEN, COLOR_BLUE, COLOR_CYAN, COLOR_YELLOW, COLOR_RED,
    COLOR_WHITE, COLOR_BLACK, COLOR_ACCENT,
    COLOR_CLOSE, COLOR_MEDIUM, COLOR_FAR,
    FONT, FONT_SCALE_TITLE, FONT_SCALE_NORMAL, FONT_SCALE_LABEL,
    FONT_SCALE_SMALL, FONT_THICKNESS, FONT_THICKNESS_BOLD, LINE_TYPE,
    HUD_TOP_BAR_HEIGHT, HUD_BOTTOM_BAR_HEIGHT, HUD_OVERLAY_ALPHA,
    CORNER_ACCENT_LENGTH, RECT_THICKNESS, CORNER_THICKNESS,
    HUD_TITLE_X, HUD_TITLE_Y, HUD_FACECOUNT_X, HUD_FPS_X, HUD_MODE_X,
    HUD_CONTROLS_Y_OFFSET, RECORDING_INDICATOR_RADIUS,
    RECORDING_INDICATOR_X_OFFSET,
    LABEL_PADDING_X, LABEL_PADDING_Y, LABEL_BG_PADDING,
    SCREENSHOT_PREFIX, SCREENSHOT_FORMAT, SCREENSHOT_TIMESTAMP_FMT,
    SCREENSHOT_FLASH_FRAMES,
    DEFAULT_DETECT_EYES, DEFAULT_DETECT_SMILES, DEFAULT_FLIP_CAMERA,
    KEY_QUIT, KEY_SCREENSHOT, KEY_TOGGLE_EYES, KEY_TOGGLE_SMILE,
    KEY_FLIP_CAMERA,
    CONTROLS_TEXT, SMILE_LABEL_TEXT, SMILE_LABEL_Y_OFFSET,
)

# load classifiers
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
smile_cascade = cv2.CascadeClassifier(SMILE_CASCADE_PATH)

if face_cascade.empty():
    print("[ERROR] Could not load face cascade classifier.")
    print("Make sure OpenCV is installed: pip install opencv-python")
    exit(1)


def create_screenshot_dir():
    if not os.path.exists(SCREENSHOT_DIR):
        os.makedirs(SCREENSHOT_DIR)
        print(f"[INFO] Created screenshot directory: {SCREENSHOT_DIR}")


def draw_fancy_rectangle(frame, x, y, w, h, color, thickness=RECT_THICKNESS,
                         corner_length=CORNER_ACCENT_LENGTH):
    # thin main rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, FONT_THICKNESS)

    # corner accents
    cv2.line(frame, (x, y), (x + corner_length, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + corner_length), color, thickness)

    cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, thickness)

    cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, thickness)

    cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, thickness)


def draw_label(frame, text, x, y, bg_color, text_color=COLOR_WHITE):
    (text_w, text_h), baseline = cv2.getTextSize(
        text, FONT, FONT_SCALE_LABEL, FONT_THICKNESS
    )
    cv2.rectangle(
        frame,
        (x, y - text_h - LABEL_BG_PADDING),
        (x + text_w + LABEL_BG_PADDING, y),
        bg_color, -1
    )
    cv2.putText(
        frame, text,
        (x + LABEL_PADDING_X, y - LABEL_PADDING_Y),
        FONT, FONT_SCALE_LABEL, text_color, FONT_THICKNESS, LINE_TYPE
    )


def draw_hud(frame, face_count, fps, mode_text, recording=False):
    h, w = frame.shape[:2]

    # top bar overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, HUD_TOP_BAR_HEIGHT), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, HUD_OVERLAY_ALPHA, frame,
                    1 - HUD_OVERLAY_ALPHA, 0, frame)

    cv2.putText(frame, APP_NAME,
                (HUD_TITLE_X, HUD_TITLE_Y),
                FONT, FONT_SCALE_TITLE, COLOR_ACCENT, FONT_THICKNESS_BOLD, LINE_TYPE)

    cv2.putText(frame, f"Faces: {face_count}",
                (HUD_FACECOUNT_X, HUD_TITLE_Y),
                FONT, FONT_SCALE_NORMAL, COLOR_GREEN, FONT_THICKNESS, LINE_TYPE)

    cv2.putText(frame, f"FPS: {fps:.0f}",
                (HUD_FPS_X, HUD_TITLE_Y),
                FONT, FONT_SCALE_NORMAL, COLOR_CYAN, FONT_THICKNESS, LINE_TYPE)

    cv2.putText(frame, mode_text,
                (HUD_MODE_X, HUD_TITLE_Y),
                FONT, FONT_SCALE_LABEL, COLOR_YELLOW, FONT_THICKNESS, LINE_TYPE)

    # bottom bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - HUD_BOTTOM_BAR_HEIGHT), (w, h),
                  COLOR_BLACK, -1)
    cv2.addWeighted(overlay2, HUD_OVERLAY_ALPHA, frame,
                    1 - HUD_OVERLAY_ALPHA, 0, frame)

    cv2.putText(frame, CONTROLS_TEXT,
                (HUD_TITLE_X, h - HUD_CONTROLS_Y_OFFSET),
                FONT, FONT_SCALE_SMALL, COLOR_WHITE, FONT_THICKNESS, LINE_TYPE)

    if recording:
        cv2.circle(frame,
                   (w - RECORDING_INDICATOR_X_OFFSET, HUD_TITLE_Y - 7),
                   RECORDING_INDICATOR_RADIUS, COLOR_RED, -1)


def detect_and_draw(frame, gray, detect_eyes=True, detect_smiles=False):
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE
    )

    face_count = len(faces)

    for i, (x, y, w, h) in enumerate(faces):
        # pick color based on how close the face is
        area = w * h
        if area > FACE_AREA_CLOSE:
            color = COLOR_CLOSE
        elif area > FACE_AREA_MEDIUM:
            color = COLOR_MEDIUM
        else:
            color = COLOR_FAR

        draw_fancy_rectangle(frame, x, y, w, h, color, thickness=CORNER_THICKNESS)

        label = f"Face #{i + 1}"
        draw_label(frame, label, x, y, color)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        if detect_eyes:
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=EYE_SCALE_FACTOR,
                minNeighbors=EYE_MIN_NEIGHBORS,
                minSize=EYE_MIN_SIZE
            )
            for (ex, ey, ew, eh) in eyes[:EYE_MAX_COUNT]:
                center = (x + ex + ew // 2, y + ey + eh // 2)
                radius = min(ew, eh) // 2
                cv2.circle(frame, center, radius, COLOR_BLUE, FONT_THICKNESS)

        if detect_smiles:
            smiles = smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=SMILE_SCALE_FACTOR,
                minNeighbors=SMILE_MIN_NEIGHBORS,
                minSize=SMILE_MIN_SIZE
            )
            if len(smiles) > 0:
                draw_label(frame, SMILE_LABEL_TEXT,
                           x, y + h + SMILE_LABEL_Y_OFFSET, COLOR_GREEN)

    return face_count


def save_screenshot(frame):
    create_screenshot_dir()
    timestamp = datetime.datetime.now().strftime(SCREENSHOT_TIMESTAMP_FMT)
    filename = os.path.join(
        SCREENSHOT_DIR,
        f"{SCREENSHOT_PREFIX}_{timestamp}{SCREENSHOT_FORMAT}"
    )
    cv2.imwrite(filename, frame)
    print(f"[INFO] Screenshot saved: {filename}")
    return filename


def main():
    print("=" * 55)
    print(f"  {APP_NAME} - Real-Time Face Detection System")
    print(f"  {APP_EVENT}")
    print("=" * 55)
    print()
    print("[INFO] Starting webcam...")
    print("[INFO] Controls:")
    print("  Q - Quit")
    print("  S - Save screenshot")
    print("  E - Toggle eye detection")
    print("  M - Toggle smile detection")
    print("  F - Flip camera")
    print()

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        print("  - Check if your webcam is connected")
        print("  - Check if another app is using it")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    detect_eyes = DEFAULT_DETECT_EYES
    detect_smiles = DEFAULT_DETECT_SMILES
    flip_camera = DEFAULT_FLIP_CAMERA
    prev_time = time.time()
    fps = 0
    total_faces_detected = 0
    frame_count = 0
    screenshot_flash = 0

    print("[INFO] Webcam started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        if flip_camera:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        frame_count += 1

        modes = []
        if detect_eyes:
            modes.append("Eyes")
        if detect_smiles:
            modes.append("Smile")
        mode_text = "Mode: " + ("+".join(modes) if modes else "Face Only")

        face_count = detect_and_draw(frame, gray, detect_eyes, detect_smiles)
        total_faces_detected += face_count

        draw_hud(frame, face_count, fps, mode_text)

        if screenshot_flash > 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0),
                          (frame.shape[1], frame.shape[0]), COLOR_WHITE, -1)
            alpha = screenshot_flash / float(SCREENSHOT_FLASH_FRAMES)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            screenshot_flash -= 1

        cv2.imshow(APP_TITLE, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == KEY_QUIT or key == KEY_QUIT - 32:
            print("\n[INFO] Shutting down...")
            break
        elif key == KEY_SCREENSHOT or key == KEY_SCREENSHOT - 32:
            save_screenshot(frame)
            screenshot_flash = SCREENSHOT_FLASH_FRAMES
        elif key == KEY_TOGGLE_EYES or key == KEY_TOGGLE_EYES - 32:
            detect_eyes = not detect_eyes
            print(f"[INFO] Eye detection: {'ON' if detect_eyes else 'OFF'}")
        elif key == KEY_TOGGLE_SMILE or key == KEY_TOGGLE_SMILE - 32:
            detect_smiles = not detect_smiles
            print(f"[INFO] Smile detection: {'ON' if detect_smiles else 'OFF'}")
        elif key == KEY_FLIP_CAMERA or key == KEY_FLIP_CAMERA - 32:
            flip_camera = not flip_camera
            print(f"[INFO] Camera flip: {'ON' if flip_camera else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 55)
    print("  Session Statistics")
    print("=" * 55)
    print(f"  Total frames processed: {frame_count}")
    print(f"  Average FPS: {frame_count / max(1, time.time() - prev_time):.1f}")
    print(f"  Total face detections: {total_faces_detected}")
    print("=" * 55)
    print(f"  Thank you for using {APP_NAME}!")
    print("=" * 55)


if __name__ == "__main__":
    main()
