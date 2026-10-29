
"""
Experiment 10: Synthetic Canvas with Four Corner Boxes (1/10th Dimension)
Objective: Generate a blank white canvas of user-specified dimensions (W x H).
Draw four distinctly colored rectangles at the four corners, each occupying
exactly one-tenth of the width and height.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_corner_boxes_canvas(width=600, height=400):
    # Create white canvas (BGR)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    # 1/10th dimensions
    bw = width // 10
    bh = height // 10

    # Define 4 distinct colors (BGR format)
    colors = {
        "Top-Left (Red)": (0, 0, 220),
        "Top-Right (Green)": (0, 180, 0),
        "Bottom-Left (Blue)": (220, 0, 0),
        "Bottom-Right (Orange)": (0, 140, 255)
    }

    # 1. Top-Left Box
    cv2.rectangle(canvas, (0, 0), (bw, bh), colors["Top-Left (Red)"], -1)
    # 2. Top-Right Box
    cv2.rectangle(canvas, (width - bw, 0), (width, bh), colors["Top-Right (Green)"], -1)
    # 3. Bottom-Left Box
    cv2.rectangle(canvas, (0, height - bh), (bw, height), colors["Bottom-Left (Blue)"], -1)
    # 4. Bottom-Right Box
    cv2.rectangle(canvas, (width - bw, height - bh), (width, height), colors["Bottom-Right (Orange)"], -1)

    # Add descriptive center text
    info = f"Canvas: {width}x{height} | Box Size: {bw}x{bh} (1/10th)"
    cv2.putText(canvas, info, (width // 2 - 180, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

    # Convert to RGB for matplotlib rendering
    rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.imshow(rgb_canvas)
    ax.set_title(f"Exp 10: Four Corner Rectangles ({bw}x{bh}) on Canvas ({width}x{height})",
                 fontsize=12, fontweight='bold')
    ax.set_xticks(range(0, width + 1, 100))
    ax.set_yticks(range(0, height + 1, 100))
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp10] Output saved at: {output_path}")

if __name__ == "__main__":
    generate_corner_boxes_canvas(width=600, height=400)
