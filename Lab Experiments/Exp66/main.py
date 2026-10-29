"""
Experiment 66: Vehicle Detection Across Video Frames
Objective: Identify moving vehicles across successive video frames,
render bounding rectangles, and ensure consistent vehicle localization.
"""

import os
import cv2
import matplotlib.pyplot as plt

def track_vehicles(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25, detectShadows=False)

    key_frames = []
    frame_indices = [10, 20, 30, 40]
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated = frame.copy()

        for cnt in contours:
            if cv2.contourArea(cnt) > 350:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(annotated, "Vehicle", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if frame_idx in frame_indices:
            key_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        frame_idx += 1
    cap.release()

    fig, axes = plt.subplots(1, len(key_frames), figsize=(16, 4))
    for ax, idx, f in zip(axes, frame_indices, key_frames):
        ax.imshow(f)
        ax.set_title(f"Vehicle Detection (Frame {idx})", fontweight='bold', fontsize=10)
        ax.axis('off')

    plt.suptitle("Exp 66: Successive Frame Vehicle Detection & Tracking", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(video_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp66] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    track_vehicles(os.path.join(script_dir, "input_video.mp4"))
