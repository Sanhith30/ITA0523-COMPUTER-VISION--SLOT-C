"""
Experiment 56: Synthetic Canvas with 4 Proportional Corner Rectangles
Objective: Create white canvas of user-specified width and height,
draw four colored boxes at corners occupying 1/10th of corresponding dimensions.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_four_corner_canvas(width=800, height=500):
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    bw = width // 10
    bh = height // 10

    # Draw 4 colored corner boxes
    cv2.rectangle(canvas, (0, 0), (bw, bh), (0, 0, 255), -1)              # Top-Left Red
    cv2.rectangle(canvas, (width - bw, 0), (width, bh), (0, 200, 0), -1)  # Top-Right Green
    cv2.rectangle(canvas, (0, height - bh), (bw, height), (255, 0, 0), -1)# Bottom-Left Blue
    cv2.rectangle(canvas, (width - bw, height - bh), (width, height), (0, 140, 255), -1) # Bottom-Right Orange

    # Labels
    cv2.putText(canvas, f"Canvas: {width}x{height}", (width//2 - 130, height//2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
    cv2.putText(canvas, f"Box Size: {bw}x{bh} (1/10th)", (width//2 - 145, height//2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cv2.imwrite(os.path.join(script_dir, "canvas_corners.jpg"), canvas)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Exp 56: Corner Rectangles ({bw}x{bh}) on Canvas ({width}x{height})", fontweight='bold')
    ax.axis('off')

    output_path = os.path.join(script_dir, "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp56] Output saved at: {output_path}")

if __name__ == "__main__":
    generate_four_corner_canvas(width=800, height=500)
