"""
Experiment 63: Video Playback Speed Adjustment
Objective: Generate slow-motion (0.5x) and accelerated (2.0x) video files,
evaluating playback continuity and temporal pacing.
"""

import os
import cv2
import matplotlib.pyplot as plt

def modify_playback_speeds(input_path):
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

    # Slow motion
    out_slow = cv2.VideoWriter(os.path.join(script_dir, "slow_motion.mp4"), fourcc, fps * 0.5, (w, h))
    for f in frames:
        out_slow.write(f)
    out_slow.release()

    # Fast motion
    out_fast = cv2.VideoWriter(os.path.join(script_dir, "fast_motion.mp4"), fourcc, fps * 2.0, (w, h))
    for f in frames:
        out_fast.write(f)
    out_fast.release()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sample = cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2RGB)

    axes[0].imshow(sample)
    axes[0].set_title(f"Original Video\n({fps:.0f} FPS, {len(frames)/fps:.1f}s)", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(sample)
    axes[1].set_title(f"Slow Motion (0.5x)\n({fps*0.5:.0f} FPS, Smooth Timing)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(sample)
    axes[2].set_title(f"Fast Motion (2.0x)\n({fps*2.0:.0f} FPS, Accelerated)", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 63: Video Playback Timing & Speed Variation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(script_dir, "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp63] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    modify_playback_speeds(os.path.join(script_dir, "input_video.mp4"))
