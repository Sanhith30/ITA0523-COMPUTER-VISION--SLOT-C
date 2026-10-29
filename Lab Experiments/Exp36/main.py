"""
Experiment 36: Affine Transformation Matrix Mapping
Objective: Compute the 2x3 affine matrix from 3 control points, warp the image,
and evaluate scale, rotation, and shear distortion.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def affine_transform_3pts(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    pts1 = np.float32([[100, 100], [w - 100, 100], [100, h - 100]])
    pts2 = np.float32([[150, 150], [w - 50, 80], [80, h - 50]])

    M = cv2.getAffineTransform(pts1, pts2)
    warped = cv2.warpAffine(rgb, M, (w, h))

    # Annotate points
    annotated_src = rgb.copy()
    annotated_dst = warped.copy()
    for pt in pts1:
        cv2.circle(annotated_src, tuple(pt.astype(int)), 12, (255, 0, 0), -1)
    for pt in pts2:
        cv2.circle(annotated_dst, tuple(pt.astype(int)), 12, (0, 255, 0), -1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(annotated_src)
    axes[0].set_title("Source Points (Blue)", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(annotated_dst)
    axes[1].set_title("Affine Warped (Destination Points Green)", fontweight='bold')
    axes[1].axis('off')

    plt.suptitle("Exp 36: Affine Transformation via 3-Point Homography", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp36] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    affine_transform_3pts(os.path.join(script_dir, "input.jpg"))
