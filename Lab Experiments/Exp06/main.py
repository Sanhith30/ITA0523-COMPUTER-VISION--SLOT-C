"""
Experiment 06: Video Frame Sequence Reversal
Objective: Read an input video, reverse the chronological order of its frames,
and write out a reversed video while preserving the original frame rate and dimensions.
"""

import os
import cv2
import matplotlib.pyplot as plt

def reverse_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        raise ValueError("No frames extracted from input video.")

    # Write reversed frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for f in reversed(frames):
        out.write(f)
    out.release()
    print(f"[Exp06] Reversed {total_frames} frames to {output_path} at {fps} FPS")

    # Generate output screenshot showing keyframe progression
    key_indices = [0, total_frames // 3, 2 * total_frames // 3, total_frames - 1]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    for col, idx in enumerate(key_indices):
        # Forward frame
        f_orig = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        axes[0, col].imshow(f_orig)
        axes[0, col].set_title(f"Original Frame #{idx+1}")
        axes[0, col].axis('off')

        # Reversed frame (index from end)
        f_rev = cv2.cvtColor(frames[total_frames - 1 - idx], cv2.COLOR_BGR2RGB)
        axes[1, col].imshow(f_rev)
        axes[1, col].set_title(f"Reversed Frame #{idx+1}")
        axes[1, col].axis('off')

    plt.suptitle("Exp 06: Video Chronological Reversal Progression", fontsize=14, fontweight='bold')
    plt.tight_layout()
    screenshot_path = os.path.join(os.path.dirname(input_path), "output.png")
    plt.savefig(screenshot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp06] Output screenshot saved at: {screenshot_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reverse_video(
        os.path.join(script_dir, "input_video.mp4"),
        os.path.join(script_dir, "output_reversed.mp4")
    )
