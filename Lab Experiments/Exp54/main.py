"""
Experiment 54: Reverse Video Frame Sequence
Objective: Buffer video frames, write them sequentially from the last frame
to the first frame into a new file, and verify smooth playback.
"""

import os
import cv2
import matplotlib.pyplot as plt

def reverse_frames(input_path):
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

    script_dir = os.path.dirname(input_path)
    out_path = os.path.join(script_dir, "reversed_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for f in reversed(frames):
        out.write(f)
    out.release()

    total = len(frames)
    indices = [0, total // 3, 2 * total // 3, total - 1]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))

    for col, idx in enumerate(indices):
        f_orig = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        axes[0, col].imshow(f_orig)
        axes[0, col].set_title(f"Original Frame #{idx+1}")
        axes[0, col].axis('off')

        f_rev = cv2.cvtColor(frames[total - 1 - idx], cv2.COLOR_BGR2RGB)
        axes[1, col].imshow(f_rev)
        axes[1, col].set_title(f"Reversed Frame #{idx+1}")
        axes[1, col].axis('off')

    plt.suptitle("Exp 54: Video Sequence Reversal Verification", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(script_dir, "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp54] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reverse_frames(os.path.join(script_dir, "input_video.mp4"))
