"""
Experiment 34: Video Frame Chronological Reversal
Objective: Read an input video, reverse the chronological frame sequence,
save the reversed video, and verify visual continuity and frame timing.
"""

import os
import cv2
import matplotlib.pyplot as plt

def reverse_video_chronology(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total = len(frames)
    output_video_path = os.path.join(os.path.dirname(input_path), "reversed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    for f in reversed(frames):
        out.write(f)
    out.release()

    # Create montage screenshot
    indices = [0, total // 3, 2 * total // 3, total - 1]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))

    for col, idx in enumerate(indices):
        f_fwd = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        axes[0, col].imshow(f_fwd)
        axes[0, col].set_title(f"Original Frame #{idx+1}")
        axes[0, col].axis('off')

        f_rev = cv2.cvtColor(frames[total - 1 - idx], cv2.COLOR_BGR2RGB)
        axes[1, col].imshow(f_rev)
        axes[1, col].set_title(f"Reversed Frame #{idx+1}")
        axes[1, col].axis('off')

    plt.suptitle("Exp 34: Video Frame Reversal Progression", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(input_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp34] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reverse_video_chronology(os.path.join(script_dir, "input_video.mp4"))
