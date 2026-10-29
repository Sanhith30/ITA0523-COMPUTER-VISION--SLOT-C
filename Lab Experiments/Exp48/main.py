"""
Experiment 48: Video Speed Modification (Slow-Motion and Accelerated)
Objective: Read a video, modify frame timing/sampling to produce both slow-motion
and accelerated outputs, and observe temporal continuity.
"""

import os
import cv2
import matplotlib.pyplot as plt

def process_video_speeds(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

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

    script_dir = os.path.dirname(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Slow motion (0.5x FPS)
    out_slow = cv2.VideoWriter(os.path.join(script_dir, "slow_motion.mp4"), fourcc, fps * 0.5, (w, h))
    for f in frames:
        out_slow.write(f)
    out_slow.release()

    # Fast motion (2.0x FPS)
    out_fast = cv2.VideoWriter(os.path.join(script_dir, "accelerated.mp4"), fourcc, fps * 2.0, (w, h))
    for f in frames:
        out_fast.write(f)
    out_fast.release()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sample_frame = cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2RGB)

    axes[0].imshow(sample_frame)
    axes[0].set_title(f"Original Video\n({fps:.0f} FPS, {len(frames)/fps:.1f}s)", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(sample_frame)
    axes[1].set_title(f"Slow Motion (0.5x)\n({fps*0.5:.0f} FPS, Extended Timing)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(sample_frame)
    axes[2].set_title(f"Accelerated (2.0x)\n({fps*2.0:.0f} FPS, Rapid Playback)", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 48: Variable Speed Video Synthesis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(script_dir, "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp48] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    process_video_speeds(os.path.join(script_dir, "input_video.mp4"))
