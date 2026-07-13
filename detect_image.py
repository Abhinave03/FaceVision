import cv2
import os
import sys
import argparse
import glob
import numpy as np

from config import (
    FACE_CASCADE_PATH, EYE_CASCADE_PATH,
    FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS, FACE_MIN_SIZE,
    EYE_SCALE_FACTOR, EYE_MIN_NEIGHBORS, EYE_MIN_SIZE, EYE_MAX_COUNT,
    COLOR_GREEN, COLOR_BLUE, COLOR_WHITE, COLOR_BLACK,
    FONT, FONT_SCALE_NORMAL, FONT_THICKNESS, LINE_TYPE,
    CORNER_ACCENT_LENGTH, RECT_THICKNESS, CORNER_THICKNESS_IMAGE,
    LABEL_PADDING_X, LABEL_PADDING_Y, LABEL_BG_PADDING,
    HUD_OVERLAY_ALPHA, HUD_TITLE_X,
    IMAGE_WINDOW_NAME, IMAGE_MAX_DISPLAY_DIM,
    IMAGE_SUMMARY_BAR_HEIGHT, IMAGE_SUMMARY_Y_OFFSET,
    OUTPUT_DIR, SUPPORTED_IMAGE_EXTENSIONS,
    APP_NAME,
)

FACE_CASCADE = cv2.CascadeClassifier(FACE_CASCADE_PATH)
EYE_CASCADE = cv2.CascadeClassifier(EYE_CASCADE_PATH)


def detect_faces_in_image(image_path, output_path=None, show=True):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return 0

    print(f"\n[INFO] Processing: {image_path}")
    print(f"  Image size: {img.shape[1]}x{img.shape[0]}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE
    )

    face_count = len(faces)
    print(f"  Faces detected: {face_count}")

    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_GREEN, RECT_THICKNESS)

        # corner accents
        cl = CORNER_ACCENT_LENGTH
        ct = CORNER_THICKNESS_IMAGE
        cv2.line(img, (x, y), (x + cl, y), COLOR_GREEN, ct)
        cv2.line(img, (x, y), (x, y + cl), COLOR_GREEN, ct)
        cv2.line(img, (x + w, y), (x + w - cl, y), COLOR_GREEN, ct)
        cv2.line(img, (x + w, y), (x + w, y + cl), COLOR_GREEN, ct)
        cv2.line(img, (x, y + h), (x + cl, y + h), COLOR_GREEN, ct)
        cv2.line(img, (x, y + h), (x, y + h - cl), COLOR_GREEN, ct)
        cv2.line(img, (x + w, y + h), (x + w - cl, y + h), COLOR_GREEN, ct)
        cv2.line(img, (x + w, y + h), (x + w, y + h - cl), COLOR_GREEN, ct)

        # label above face
        label = f"Face #{i + 1}"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE_NORMAL, FONT_THICKNESS)
        cv2.rectangle(img, (x, y - th - LABEL_BG_PADDING),
                      (x + tw + LABEL_BG_PADDING, y), COLOR_GREEN, -1)
        cv2.putText(img, label, (x + LABEL_PADDING_X, y - LABEL_PADDING_Y),
                    FONT, FONT_SCALE_NORMAL, COLOR_WHITE, FONT_THICKNESS, LINE_TYPE)

        # detect eyes inside face region
        roi_gray = gray[y:y + h, x:x + w]
        eyes = EYE_CASCADE.detectMultiScale(
            roi_gray,
            scaleFactor=EYE_SCALE_FACTOR,
            minNeighbors=EYE_MIN_NEIGHBORS,
            minSize=EYE_MIN_SIZE
        )
        for (ex, ey, ew, eh) in eyes[:EYE_MAX_COUNT]:
            center = (x + ex + ew // 2, y + ey + eh // 2)
            radius = min(ew, eh) // 2
            cv2.circle(img, center, radius, COLOR_BLUE, RECT_THICKNESS)

        print(f"    Face #{i + 1}: pos=({x},{y}), size={w}x{h}, eyes={len(eyes[:EYE_MAX_COUNT])}")

    # summary bar at bottom
    h_img, w_img = img.shape[:2]
    summary = f"{APP_NAME} | Detected: {face_count} face(s)"
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h_img - IMAGE_SUMMARY_BAR_HEIGHT),
                  (w_img, h_img), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, HUD_OVERLAY_ALPHA, img, 1 - HUD_OVERLAY_ALPHA, 0, img)
    cv2.putText(img, summary,
                (HUD_TITLE_X, h_img - IMAGE_SUMMARY_Y_OFFSET),
                FONT, FONT_SCALE_NORMAL, COLOR_WHITE, FONT_THICKNESS, LINE_TYPE)

    # save output
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_detected{ext}"

    cv2.imwrite(output_path, img)
    print(f"  Output saved: {output_path}")

    if show:
        display = img.copy()
        if max(display.shape[:2]) > IMAGE_MAX_DISPLAY_DIM:
            scale = IMAGE_MAX_DISPLAY_DIM / max(display.shape[:2])
            display = cv2.resize(display, None, fx=scale, fy=scale)
        cv2.imshow(IMAGE_WINDOW_NAME, display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return face_count


def process_directory(directory_path, output_dir=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    image_files = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(directory_path, ext)))

    if not image_files:
        print(f"[ERROR] No images found in: {directory_path}")
        return

    print(f"[INFO] Found {len(image_files)} images to process")

    total_faces = 0
    for img_path in image_files:
        basename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"detected_{basename}")
        faces = detect_faces_in_image(img_path, output_path, show=False)
        total_faces += faces

    print(f"\n[SUMMARY] Processed {len(image_files)} images, {total_faces} faces total")
    print(f"[SUMMARY] Results saved in: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} - Face Detection on Static Images"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input image or directory")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path (single image mode)")
    parser.add_argument("--no-display", action="store_true",
                        help="Don't show result window")

    args = parser.parse_args()

    if os.path.isdir(args.input):
        process_directory(args.input)
    elif os.path.isfile(args.input):
        detect_faces_in_image(args.input, args.output, show=not args.no_display)
    else:
        print(f"[ERROR] Path not found: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
