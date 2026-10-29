"""
Experiment 18: Perspective Transformation Using 4 Control Points
Objective: Calculate the 3x3 perspective transformation matrix from 4 non-collinear
control points and apply perspective mapping using cv2.warpPerspective.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def perspective_transformation(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Select 4 quadrilateral source points
    pts1 = np.float32([
        [int(w * 0.25), int(h * 0.15)],
        [int(w * 0.75), int(h * 0.10)],
        [int(w * 0.15), int(h * 0.85)],
        [int(w * 0.85), int(h * 0.90)]
    ])

    # Target rectangular output points
    out_w, out_h = 600, 450
    pts2 = np.float32([
        [0, 0],
        [out_w - 1, 0],
        [0, out_h - 1],
        [out_w - 1, out_h - 1]
    ])

    # Compute Perspective Matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(rgb, M, (out_w, out_h))

    # Mark points on source image
    annotated = rgb.copy()
    for pt in pts1:
        cv2.circle(annotated, tuple(pt.astype(int)), 10, (255, 0, 0), -1)
    pts_poly = pts1.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated, [pts_poly], isClosed=True, color=(255, 255, 0), thickness=3)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(annotated)
    axes[0].set_title("Input Image (4 Quadrilateral Control Points)", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(warped)
    axes[1].set_title("Rectified Perspective Warped Output", fontweight='bold')
    axes[1].axis('off')

    plt.suptitle("Exp 18: Four-Point Perspective Geometric Transformation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp18] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    perspective_transformation(os.path.join(script_dir, "input.jpg"))
