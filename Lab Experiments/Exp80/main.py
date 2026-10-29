"""
Experiment 80: Video Sequence Reversal with Simultaneous Slow-Motion
Objective: Read video frames, invert chronological sequence, and write them
at half the original frame rate (0.5x FPS) to produce reversed slow-motion playback.
"""

import os
import cv2
import matplotlib.pyplot as plt

def reverse_slow_motion_video(input_path):
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
    out_path = os.path.join(script_dir, "reversed_slowmo.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Half frame rate = slow motion
    slow_fps = max(5.0, fps * 0.5)
    out = cv2.VideoWriter(out_path, fourcc, slow_fps, (w, h))

    for f in reversed(frames):
        out.write(f)
    out.release()
    print(f"[Exp80] Created reversed slow-motion video ({slow_fps} FPS) at: {out_path}")

    total = len(frames)
    indices = [0, total // 3, 2 * total // 3, total - 1]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))

    for col, idx in enumerate(indices):
        f_orig = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        axes[0, col].imshow(f_orig)
        axes[0, col].set_title(f"Original Frame #{idx+1} ({fps:.0f} FPS)")
        axes[0, col].axis('off')

        f_rev = cv2.cvtColor(frames[total - 1 - idx], cv2.COLOR_BGR2RGB)
        axes[1, col].imshow(f_rev)
        axes[1, col].set_title(f"Reversed Frame #{idx+1} ({slow_fps:.0f} FPS)")
        axes[1, col].axis('off')

    plt.suptitle("Exp 80: Combined Video Reversal & Slow-Motion Playback", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(script_dir, "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp80] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reverse_slow_motion_video(os.path.join(script_dir, "input_video.mp4"))
