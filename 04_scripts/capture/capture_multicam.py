"""
T5.B.4: Dual-Camera Synchronized Capture Software.

Features:
  1. Live split-screen GUI preview for CAM01 (Frontal) + CAM02 (Lateral).
  2. Multi-threaded frame grabbing to minimize inter-camera latency.
  3. Single-trigger synchronized capture (Spacebar / Timer Countdown).
  4. Auto-naming files per convention: {subject_id}_{session_id}_{capture_id}_{camera_id}.jpg
  5. Real-time logging to captures.csv and images.csv.
  6. Posture class selection via keyboard shortcuts for efficient labeling.
  7. Decoupled calibration_id linking per rig configuration.

Usage:
  python 04_scripts/capture/capture_multicam.py \
      --subject_id S001 --session_id SE01 --calibration_id CAL_001 \
      --cam01_idx 0 --cam02_idx 1

Controls:
  [SPACE]       : Capture current frame pair (same capture_id)
  [1]-[7]       : Select posture class (1=upright, 2=lean_fwd, 3=lean_bwd, 4=lean_L, 5=lean_R, 6=slouch, 7=fwd_head)
  [0]           : Select reject class (standing/transition/no_person)
  [+] / [-]     : Increment / Decrement repetition counter
  [R]           : Reset repetition counter to 1
  [T]           : Toggle 3-second countdown timer mode
  [Q] / [ESC]   : Quit and finalize session
"""
import os
import cv2
import csv
import time
import argparse
import threading
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "02_data" / "private_raw"
META_DIR = PROJECT_ROOT / "03_metadata" / "private_templates"

# Draft posture class list (DRAFT status - subject to pilot study validation)
POSTURE_CLASSES = {
    ord('1'): "upright",
    ord('2'): "leaning_forward",
    ord('3'): "leaning_backward",
    ord('4'): "leaning_left",
    ord('5'): "leaning_right",
    ord('6'): "slouching",
    ord('7'): "forward_head",
    ord('0'): "reject",
}

POSTURE_LABELS_DISPLAY = {
    "upright":          "[1] Upright (Tegak)",
    "leaning_forward":  "[2] Lean Forward",
    "leaning_backward": "[3] Lean Backward",
    "leaning_left":     "[4] Lean Left",
    "leaning_right":    "[5] Lean Right",
    "slouching":        "[6] Slouching",
    "forward_head":     "[7] Forward Head",
    "reject":           "[0] Reject / Exclusion",
}


class ThreadedCamera:
    """Thread-safe camera reader for low-latency dual-camera capture."""
    
    def __init__(self, cam_idx, camera_id):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id} at index {cam_idx}")
        
        # Try to set resolution to 1920x1080
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        self.frame = None
        self.timestamp = None
        self.lock = threading.Lock()
        self.running = True
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
    
    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.timestamp = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
            else:
                time.sleep(0.001)
    
    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None, self.timestamp
    
    def get_resolution(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h
    
    def release(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()


def compute_blur_score(frame):
    """Compute Laplacian variance as blur metric. Higher = sharper."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)


def get_next_capture_id(captures_csv_path):
    """Read existing captures.csv and return next incremented capture_id."""
    max_id = 0
    if captures_csv_path.exists():
        with open(captures_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cap_id = row.get("capture_id", "CAP000000")
                num = int(cap_id.replace("CAP", ""))
                max_id = max(max_id, num)
    return max_id + 1


def append_capture_record(captures_csv, record):
    """Append a capture record to captures.csv."""
    fieldnames = [
        "capture_id", "subject_id", "session_id", "calibration_id",
        "primary_posture", "head_state", "shoulder_state", "pelvis_state",
        "repetition", "subset", "quality", "notes"
    ]
    write_header = not captures_csv.exists()
    with open(captures_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def append_image_record(images_csv, record):
    """Append an image record to images.csv."""
    fieldnames = [
        "image_id", "capture_id", "camera_id", "image_path",
        "timestamp", "width", "height", "blur_score",
        "exposure_status", "annotation_status"
    ]
    write_header = not images_csv.exists()
    with open(images_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def draw_overlay(display, subject_id, session_id, calibration_id,
                 current_posture, repetition, capture_count, timer_mode,
                 countdown_active, countdown_remaining):
    """Draw HUD overlay on the live preview."""
    h, w = display.shape[:2]
    
    # Semi-transparent header bar
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
    
    # Session info
    info_text = f"Subject: {subject_id} | Session: {session_id} | Calibration: {calibration_id}"
    cv2.putText(display, info_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Current posture & repetition
    posture_display = POSTURE_LABELS_DISPLAY.get(current_posture, current_posture)
    cv2.putText(display, f"Posture: {posture_display}  |  Rep: {repetition}  |  Captured: {capture_count}",
                (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)
    
    # Timer mode indicator
    timer_text = "Timer: ON (3s)" if timer_mode else "Timer: OFF (Instant)"
    cv2.putText(display, timer_text, (15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)
    
    # Countdown overlay
    if countdown_active and countdown_remaining > 0:
        # Large countdown number in center
        count_text = str(int(countdown_remaining) + 1)
        text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 5, 10)[0]
        cx = (w - text_size[0]) // 2
        cy = (h + text_size[1]) // 2
        cv2.putText(display, count_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
    
    # Help footer
    footer_y = h - 15
    help_text = "[SPACE] Capture  |  [1-7] Posture  |  [0] Reject  |  [+/-] Rep  |  [T] Timer  |  [Q] Quit"
    cv2.putText(display, help_text, (15, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    
    # CAM labels
    mid_x = w // 2
    cv2.putText(display, "CAM01 (Frontal)", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(display, "CAM02 (Lateral)", (mid_x + 15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    # Divider line
    cv2.line(display, (mid_x, 90), (mid_x, h - 30), (100, 100, 100), 2)


def run_capture_session(
    subject_id="S001",
    session_id="SE01",
    calibration_id="CAL_001",
    cam01_idx=0,
    cam02_idx=1,
    preview_width=1280,
    preview_height=480
):
    # Initialize output directories
    cam01_dir = RAW_DIR / subject_id / session_id / "CAM01"
    cam02_dir = RAW_DIR / subject_id / session_id / "CAM02"
    cam01_dir.mkdir(parents=True, exist_ok=True)
    cam02_dir.mkdir(parents=True, exist_ok=True)
    
    captures_csv = META_DIR / "captures.csv"
    images_csv = META_DIR / "images.csv"
    
    # Initialize threaded cameras
    print(f"\n{'='*60}")
    print(f"  DUAL-CAMERA CAPTURE SESSION")
    print(f"  Subject: {subject_id}  |  Session: {session_id}  |  Calibration: {calibration_id}")
    print(f"{'='*60}")
    
    try:
        cam01 = ThreadedCamera(cam01_idx, "CAM01")
        w1, h1 = cam01.get_resolution()
        print(f"  [OK] CAM01 (Frontal)  : Device {cam01_idx} -> {w1}x{h1}")
    except RuntimeError as e:
        print(f"  [ERROR] {e}")
        return
    
    try:
        cam02 = ThreadedCamera(cam02_idx, "CAM02")
        w2, h2 = cam02.get_resolution()
        print(f"  [OK] CAM02 (Lateral)  : Device {cam02_idx} -> {w2}x{h2}")
    except RuntimeError as e:
        print(f"  [ERROR] {e}")
        cam01.release()
        return
    
    # State variables
    capture_counter = get_next_capture_id(captures_csv)
    current_posture = "upright"
    repetition = 1
    capture_count = 0
    timer_mode = False
    countdown_active = False
    countdown_start = 0
    countdown_duration = 3.0
    
    half_w = preview_width // 2
    half_h = preview_height
    
    print("\n  Starting live preview... Press [Q] or [ESC] to quit.\n")
    
    while True:
        frame1, ts1 = cam01.read()
        frame2, ts2 = cam02.read()
        
        if frame1 is None or frame2 is None:
            time.sleep(0.01)
            continue
        
        # Resize for display
        disp1 = cv2.resize(frame1, (half_w, half_h))
        disp2 = cv2.resize(frame2, (half_w, half_h))
        display = np.hstack((disp1, disp2))
        
        # Handle countdown
        countdown_remaining = 0
        if countdown_active:
            elapsed = time.time() - countdown_start
            countdown_remaining = countdown_duration - elapsed
            if countdown_remaining <= 0:
                # CAPTURE NOW
                countdown_active = False
                # Trigger capture (fall through to save logic below)
                capture_triggered = True
            else:
                capture_triggered = False
        else:
            capture_triggered = False
        
        # Draw overlay
        draw_overlay(display, subject_id, session_id, calibration_id,
                     current_posture, repetition, capture_count, timer_mode,
                     countdown_active, countdown_remaining)
        
        cv2.imshow("Dual-Camera Capture", display)
        
        # Handle capture trigger (from countdown)
        if capture_triggered:
            # Re-read latest frames for actual capture
            frame1, ts1 = cam01.read()
            frame2, ts2 = cam02.read()
            if frame1 is not None and frame2 is not None:
                cap_id = f"CAP{capture_counter:06d}"
                
                # Save images
                fname1 = f"{subject_id}_{session_id}_{cap_id}_CAM01.jpg"
                fname2 = f"{subject_id}_{session_id}_{cap_id}_CAM02.jpg"
                fpath1 = cam01_dir / fname1
                fpath2 = cam02_dir / fname2
                cv2.imwrite(str(fpath1), frame1)
                cv2.imwrite(str(fpath2), frame2)
                
                blur1 = compute_blur_score(frame1)
                blur2 = compute_blur_score(frame2)
                
                # Log to captures.csv
                append_capture_record(captures_csv, {
                    "capture_id": cap_id,
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "calibration_id": calibration_id,
                    "primary_posture": current_posture,
                    "head_state": "",
                    "shoulder_state": "",
                    "pelvis_state": "",
                    "repetition": repetition,
                    "subset": "controlled",
                    "quality": "pending",
                    "notes": ""
                })
                
                # Log CAM01 to images.csv
                append_image_record(images_csv, {
                    "image_id": f"{subject_id}_{session_id}_{cap_id}_CAM01",
                    "capture_id": cap_id,
                    "camera_id": "CAM01",
                    "image_path": str(fpath1.relative_to(PROJECT_ROOT)),
                    "timestamp": ts1 or datetime.now(timezone.utc).isoformat(),
                    "width": frame1.shape[1],
                    "height": frame1.shape[0],
                    "blur_score": blur1,
                    "exposure_status": "auto",
                    "annotation_status": "unannotated"
                })
                
                # Log CAM02 to images.csv
                append_image_record(images_csv, {
                    "image_id": f"{subject_id}_{session_id}_{cap_id}_CAM02",
                    "capture_id": cap_id,
                    "camera_id": "CAM02",
                    "image_path": str(fpath2.relative_to(PROJECT_ROOT)),
                    "timestamp": ts2 or datetime.now(timezone.utc).isoformat(),
                    "width": frame2.shape[1],
                    "height": frame2.shape[0],
                    "blur_score": blur2,
                    "exposure_status": "auto",
                    "annotation_status": "unannotated"
                })
                
                capture_count += 1
                capture_counter += 1
                
                print(f"  [CAPTURED] {cap_id} | {current_posture} rep={repetition} "
                      f"| blur: CAM01={blur1:.0f} CAM02={blur2:.0f}")
        
        # Process keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key in [ord('q'), ord('Q'), 27]:  # Q or ESC
            break
        elif key == ord(' '):  # SPACE
            if timer_mode:
                if not countdown_active:
                    countdown_active = True
                    countdown_start = time.time()
            else:
                # Instant capture
                frame1, ts1 = cam01.read()
                frame2, ts2 = cam02.read()
                if frame1 is not None and frame2 is not None:
                    cap_id = f"CAP{capture_counter:06d}"
                    fname1 = f"{subject_id}_{session_id}_{cap_id}_CAM01.jpg"
                    fname2 = f"{subject_id}_{session_id}_{cap_id}_CAM02.jpg"
                    fpath1 = cam01_dir / fname1
                    fpath2 = cam02_dir / fname2
                    cv2.imwrite(str(fpath1), frame1)
                    cv2.imwrite(str(fpath2), frame2)
                    
                    blur1 = compute_blur_score(frame1)
                    blur2 = compute_blur_score(frame2)
                    
                    append_capture_record(captures_csv, {
                        "capture_id": cap_id,
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "calibration_id": calibration_id,
                        "primary_posture": current_posture,
                        "head_state": "",
                        "shoulder_state": "",
                        "pelvis_state": "",
                        "repetition": repetition,
                        "subset": "controlled",
                        "quality": "pending",
                        "notes": ""
                    })
                    append_image_record(images_csv, {
                        "image_id": f"{subject_id}_{session_id}_{cap_id}_CAM01",
                        "capture_id": cap_id,
                        "camera_id": "CAM01",
                        "image_path": str(fpath1.relative_to(PROJECT_ROOT)),
                        "timestamp": ts1 or datetime.now(timezone.utc).isoformat(),
                        "width": frame1.shape[1],
                        "height": frame1.shape[0],
                        "blur_score": blur1,
                        "exposure_status": "auto",
                        "annotation_status": "unannotated"
                    })
                    append_image_record(images_csv, {
                        "image_id": f"{subject_id}_{session_id}_{cap_id}_CAM02",
                        "capture_id": cap_id,
                        "camera_id": "CAM02",
                        "image_path": str(fpath2.relative_to(PROJECT_ROOT)),
                        "timestamp": ts2 or datetime.now(timezone.utc).isoformat(),
                        "width": frame2.shape[1],
                        "height": frame2.shape[0],
                        "blur_score": blur2,
                        "exposure_status": "auto",
                        "annotation_status": "unannotated"
                    })
                    
                    capture_count += 1
                    capture_counter += 1
                    print(f"  [CAPTURED] {cap_id} | {current_posture} rep={repetition} "
                          f"| blur: CAM01={blur1:.0f} CAM02={blur2:.0f}")
        
        elif key in POSTURE_CLASSES:
            current_posture = POSTURE_CLASSES[key]
            print(f"  >> Posture set to: {current_posture}")
        elif key == ord('+') or key == ord('='):
            repetition += 1
            print(f"  >> Repetition: {repetition}")
        elif key == ord('-') or key == ord('_'):
            repetition = max(1, repetition - 1)
            print(f"  >> Repetition: {repetition}")
        elif key in [ord('r'), ord('R')]:
            repetition = 1
            print(f"  >> Repetition reset to 1")
        elif key in [ord('t'), ord('T')]:
            timer_mode = not timer_mode
            print(f"  >> Timer mode: {'ON (3s countdown)' if timer_mode else 'OFF (instant)'}")
    
    # Cleanup
    cam01.release()
    cam02.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"  SESSION COMPLETE")
    print(f"  Total captures: {capture_count}")
    print(f"  Images saved:   {capture_count * 2} (2 cameras x {capture_count} poses)")
    print(f"  Output dirs:")
    print(f"    CAM01: {cam01_dir}")
    print(f"    CAM02: {cam02_dir}")
    print(f"  Metadata: {captures_csv}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Dual-Camera Synchronized Capture")
    parser.add_argument("--subject_id", type=str, default="S001", help="Subject ID (e.g. S001)")
    parser.add_argument("--session_id", type=str, default="SE01", help="Session ID (e.g. SE01)")
    parser.add_argument("--calibration_id", type=str, default="CAL_001", help="Calibration rig ID")
    parser.add_argument("--cam01_idx", type=int, default=0, help="Camera device index for CAM01 (Frontal)")
    parser.add_argument("--cam02_idx", type=int, default=1, help="Camera device index for CAM02 (Lateral)")
    parser.add_argument("--preview_width", type=int, default=1280, help="Total preview window width")
    parser.add_argument("--preview_height", type=int, default=480, help="Preview window height")

    args = parser.parse_args()
    run_capture_session(
        subject_id=args.subject_id,
        session_id=args.session_id,
        calibration_id=args.calibration_id,
        cam01_idx=args.cam01_idx,
        cam02_idx=args.cam02_idx,
        preview_width=args.preview_width,
        preview_height=args.preview_height
    )


if __name__ == "__main__":
    main()
