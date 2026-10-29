"""
Experiment 24: Video Playback Speed Manipulation (Slow-Motion and Fast-Motion)
Objective: Generate both slow-motion and fast-motion versions of an input video
by adjusting frame playback timing and frame sampling rates.
"""

import os
import cv2
import matplotlib.pyplot as plt

def change_video_speed(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    script_dir = os.path.dirname(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 1. Slow-Motion: half the FPS (0.5x speed)
    slow_path = os.path.join(script_dir, "output_slow.mp4")
    out_slow = cv2.VideoWriter(slow_path, fourcc, fps * 0.5, (w, h))
    for f in frames:
        out_slow.write(f)
    out_slow.release()

    # 2. Fast-Motion: double the FPS (2x speed) or skip every 2nd frame
    fast_path = os.path.join(script_dir, "output_fast.mp4")
    out_fast = cv2.VideoWriter(fast_path, fourcc, fps * 2.0, (w, h))
    for f in frames:
        out_fast.write(f)
    out_fast.release()

    print(f"[Exp24] Created slow motion ({fps*0.5} FPS) and fast motion ({fps*2.0} FPS)")

    # Output comparison screenshot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mid_idx = len(frames) // 2
    f_sample = cv2.cvtColor(frames[mid_idx], cv2.COLOR_BGR2RGB)

    axes[0].imshow(f_sample)
    axes[0].set_title(f"Original Video\n({fps:.0f} FPS, Duration: {len(frames)/fps:.1f}s)", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(f_sample)
    axes[1].set_title(f"Slow Motion (0.5x)\n({fps*0.5:.0f} FPS, Duration: {len(frames)/(fps*0.5):.1f}s)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(f_sample)
    axes[2].set_title(f"Fast Motion (2.0x)\n({fps*2.0:.0f} FPS, Duration: {len(frames)/(fps*2.0):.1f}s)", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 24: Temporal Video Speed Manipulation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(script_dir, "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp24] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    change_video_speed(os.path.join(script_dir, "input_video.mp4"))
