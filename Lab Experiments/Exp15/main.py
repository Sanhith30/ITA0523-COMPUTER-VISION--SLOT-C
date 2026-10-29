"""
Experiment 15: Frame-by-Frame Vehicle Detection in Video
Objective: Process video frames sequentially, detect moving vehicles using
Background Subtraction (MOG2) and contour bounding boxes, and verify tracking.
"""

import os
import cv2
import matplotlib.pyplot as plt

def detect_vehicles_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    # Use Background Subtractor MOG2 for vehicle localization
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=30, detectShadows=False)

    frames_tracked = []
    frame_indices = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()
        fg_mask = bg_subtractor.apply(frame)

        # Morphological filtering to eliminate noise and bridge vehicle contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vehicle_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400: # filter small false positives
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated, "Vehicle", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                vehicle_count += 1

        cv2.putText(annotated, f"Frame {frame_idx} | Detected: {vehicle_count}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if frame_idx in [10, 20, 30, 40]:
            frames_tracked.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            frame_indices.append(frame_idx)

        frame_idx += 1
    cap.release()

    # Generate output screenshot showing 4 sequential frames
    fig, axes = plt.subplots(1, len(frames_tracked), figsize=(16, 4))
    for ax, idx, f in zip(axes, frame_indices, frames_tracked):
        ax.imshow(f)
        ax.set_title(f"Sequential Detection (Frame {idx})", fontweight='bold', fontsize=10)
        ax.axis('off')

    plt.suptitle("Exp 15: Frame-by-Frame Vehicle Tracking & Localization", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(video_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp15] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    detect_vehicles_in_video(os.path.join(script_dir, "input_video.mp4"))
