import cv2
import time
import numpy as np
import os
import json
import datetime

from config import (
    FACE_CASCADE_PATH, EYE_CASCADE_PATH, SMILE_CASCADE_PATH,
    FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS, FACE_MIN_SIZE,
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    BENCHMARK_RESOLUTIONS, BENCHMARK_ITERATIONS, BENCHMARK_WEBCAM_DURATION,
    BENCHMARK_REPORT_PATH,
    APP_NAME,
)


def benchmark_detection():
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    print("=" * 55)
    print(f"  {APP_NAME} - Performance Benchmark")
    print("=" * 55)
    print()

    # test 1 - detection speed across resolutions
    print("[TEST 1] Detection Speed on Various Image Sizes")
    print("-" * 50)

    results = []

    for w, h in BENCHMARK_RESOLUTIONS:
        test_img = np.random.randint(0, 256, (h, w), dtype=np.uint8)

        times = []
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter()
            _ = face_cascade.detectMultiScale(
                test_img,
                scaleFactor=FACE_SCALE_FACTOR,
                minNeighbors=FACE_MIN_NEIGHBORS,
                minSize=FACE_MIN_SIZE
            )
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)
        potential_fps = 1000.0 / avg_time if avg_time > 0 else 0

        result = {
            "resolution": f"{w}x{h}",
            "avg_ms": round(avg_time, 2),
            "std_ms": round(std_time, 2),
            "potential_fps": round(potential_fps, 1)
        }
        results.append(result)
        print(f"  {w}x{h}: {avg_time:.2f}ms +/- {std_time:.2f}ms ({potential_fps:.1f} FPS)")

    # test 2 - webcam fps
    print()
    print(f"[TEST 2] Live Webcam FPS Test ({BENCHMARK_WEBCAM_DURATION}s)")
    print("-" * 50)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    webcam_results = {}

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        frame_count = 0
        face_detection_count = 0
        start_time = time.time()

        while (time.time() - start_time) < BENCHMARK_WEBCAM_DURATION:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_SCALE_FACTOR,
                minNeighbors=FACE_MIN_NEIGHBORS,
                minSize=FACE_MIN_SIZE
            )

            frame_count += 1
            face_detection_count += len(faces)

        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0

        webcam_results = {
            "frames_processed": frame_count,
            "duration_seconds": round(elapsed, 2),
            "average_fps": round(avg_fps, 1),
            "total_detections": face_detection_count,
            "avg_faces_per_frame": round(face_detection_count / max(1, frame_count), 2)
        }

        print(f"  Frames: {frame_count}")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Avg FPS: {avg_fps:.1f}")
        print(f"  Total detections: {face_detection_count}")
        print(f"  Avg faces/frame: {face_detection_count / max(1, frame_count):.2f}")

        cap.release()
    else:
        print("  [SKIP] Webcam not available")
        webcam_results = {"status": "webcam_unavailable"}

    # test 3 - system info
    print()
    print("[TEST 3] System Info")
    print("-" * 50)

    system_info = {
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "timestamp": datetime.datetime.now().isoformat(),
        "classifiers": {
            "face": os.path.basename(FACE_CASCADE_PATH),
            "eye": os.path.basename(EYE_CASCADE_PATH),
            "smile": os.path.basename(SMILE_CASCADE_PATH),
        }
    }

    print(f"  OpenCV: {cv2.__version__}")
    print(f"  NumPy: {np.__version__}")
    print(f"  Python: {os.sys.version}")

    report = {
        "benchmark_date": datetime.datetime.now().isoformat(),
        "detection_speed": results,
        "webcam_test": webcam_results,
        "system_info": system_info
    }

    with open(BENCHMARK_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print(f"[INFO] Report saved to: {BENCHMARK_REPORT_PATH}")
    print()
    print("=" * 55)
    print("  Benchmark Complete!")
    print("=" * 55)

    return report


if __name__ == "__main__":
    benchmark_detection()
