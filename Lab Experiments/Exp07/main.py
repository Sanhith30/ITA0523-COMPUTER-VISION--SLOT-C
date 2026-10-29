"""
Experiment 07: Affine Transformation Using 3 Control Points
Objective: Calculate the 2x3 affine transformation matrix from three corresponding
control point pairs and apply it using cv2.warpAffine to observe scale, shear, and rotation.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def perform_affine_transform(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Select 3 non-collinear source points
    pts1 = np.float32([[50, 50], [w - 100, 50], [50, h - 80]])
    # Target transformed coordinates (sheared and shifted)
    pts2 = np.float32([[80, 100], [w - 50, 80], [120, h - 30]])

    # Calculate Affine Matrix M
    M = cv2.getAffineTransform(pts1, pts2)

    # Warp image
    warped = cv2.warpAffine(rgb, M, (w, h))

    # Draw control points on copies for visualization
    vis_orig = rgb.copy()
    vis_warped = warped.copy()
    for pt in pts1:
        cv2.circle(vis_orig, tuple(pt.astype(int)), 10, (255, 0, 0), -1)
    for pt in pts2:
        cv2.circle(vis_warped, tuple(pt.astype(int)), 10, (0, 255, 0), -1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(vis_orig)
    axes[0].set_title("Original Image (with Source Control Points)", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(vis_warped)
    axes[1].set_title("Affine Transformed Output (with Destination Points)", fontweight='bold')
    axes[1].axis('off')

    plt.suptitle("Exp 07: Affine Geometric Transformation (3 Control Points)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp07] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    perform_affine_transform(os.path.join(script_dir, "input.jpg"))
