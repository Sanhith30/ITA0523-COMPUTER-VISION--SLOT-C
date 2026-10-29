"""
Experiment 46: Affine vs Perspective Transformation Comparative Study
Objective: Demonstrate 3-point affine transformation and 4-point perspective
transformation side-by-side, analyzing shearing vs projective distortion.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_affine_perspective(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # 1. Affine (3 points)
    pts_aff_src = np.float32([[50, 50], [w - 50, 50], [50, h - 50]])
    pts_aff_dst = np.float32([[100, 80], [w - 30, 100], [80, h - 30]])
    M_aff = cv2.getAffineTransform(pts_aff_src, pts_aff_dst)
    warped_affine = cv2.warpAffine(rgb, M_aff, (w, h))

    # 2. Perspective (4 points)
    pts_persp_src = np.float32([[100, 80], [w - 100, 80], [50, h - 60], [w - 50, h - 60]])
    pts_persp_dst = np.float32([[0, 0], [w, 0], [150, h], [w - 150, h]])
    M_persp = cv2.getPerspectiveTransform(pts_persp_src, pts_persp_dst)
    warped_perspective = cv2.warpPerspective(rgb, M_persp, (w, h))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Input Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(warped_affine)
    axes[1].set_title("Affine Transformed (3 Points)\n(Preserves Parallel Lines)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(warped_perspective)
    axes[2].set_title("Perspective Transformed (4 Points)\n(Simulates Camera Viewpoint / Depth)", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 46: Geometric Warping: Affine vs Perspective Homography", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp46] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    compare_affine_perspective(os.path.join(script_dir, "input.jpg"))
