# src/align.py
"""
Alignment demo using your WORKING pipeline:
- Haar face detection (fast)
- MediaPipe FaceMesh -> 5 keypoints (stable)

This avoids the bug in haar_5pt.py where the aligned window was shown
only after the loop and using stale variables.

Run:
python -m src.align

Keys:
q quit
s save current aligned face to data/debug_aligned/<timestamp>.jpg
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np

# Import from your existing script
from .haar_5pt import Haar5ptDetector, align_face_5pt

def _put_text(img, text: str, xy=(10, 30), scale=0.8, thickness=2):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def _safe_imshow(win: str, img: np.ndarray):
    if img is None:
        return
    cv2.imshow(win, img)  # Fixed: properly indented


def main(
    cam_index: int = 0,
    out_size: Tuple[int, int] = (112, 112),
    mirror: bool = True,
):
    camera_capture = cv2.VideoCapture(cam_index)
    detector = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.80,
        debug=True,
    )
    output_width, output_height = int(out_size[0]), int(out_size[1])
    blank_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    # Where to save aligned snapshots
    save_directory = Path("data/debug_aligned")
    save_directory.mkdir(parents=True, exist_ok=True)
    last_aligned_face = blank_image.copy()
    fps_start_time = time.time()
    frame_count = 0
    frames_per_second = 0.0
    print("align running. Press 'q' to quit, 's' to save aligned face.")
    while True:
        success, current_frame = camera_capture.read()
        if not success:
            break
        if mirror:
            current_frame = cv2.flip(current_frame, 1)
        
        detected_faces = detector.detect(current_frame, max_faces=1)
        
        visualization_frame = current_frame.copy()
        aligned_face = None
        
        if detected_faces:
            face = detected_faces[0]
            
            # Draw box + 5 pts
            cv2.rectangle(visualization_frame, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
            for (x, y) in face.kps.astype(int):
                cv2.circle(visualization_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                
            # Align (this is the whole point)
            aligned_face, _M = align_face_5pt(current_frame, face.kps, out_size=out_size)
            
            # Keep last good aligned (so window doesn't go black on brief misses)
            if aligned_face is not None and aligned_face.size:
                last_aligned_face = aligned_face
            _put_text(visualization_frame, "OK (Haar + FaceMesh 5pt)", (10, 30), 0.75, 2)
        else:
            _put_text(visualization_frame, "no face", (10, 30), 0.9, 2)
        
        # FPS
        frame_count += 1
        dt = time.time() - fps_start_time
        if dt >= 1.0:
            frames_per_second = frame_count / dt
            frame_count = 0
            fps_start_time = time.time()
        
        _put_text(visualization_frame, f"FPS: {frames_per_second:.1f}", (10, 60), 0.75, 2)
        _put_text(visualization_frame, f"warp: 5pt -> {output_width}x{output_height}", (10, 90), 0.75, 2)
        
        _safe_imshow("align - camera", visualization_frame)
        _safe_imshow("align - aligned", last_aligned_face)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            ts = int(time.time() * 1000)
            out_path = save_directory / f"{ts}.jpg"
            cv2.imwrite(str(out_path), last_aligned_face)
            print(f"[align] saved: {out_path}")
            
    camera_capture.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()