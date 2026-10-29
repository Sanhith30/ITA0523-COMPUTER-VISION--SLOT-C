"""
Experiment 60: Affine and Perspective Transformations
Objective: Demonstrate 3-point affine mapping and 4-point perspective mapping,
comparing alignment, shear, scale, orientation, and perspective distortion.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_transformations(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Affine (3 points)
    p_aff_src = np.float32([[50, 50], [w - 80, 50], [50, h - 80]])
    p_aff_dst = np.float32([[90, 100], [w - 40, 80], [120, h - 30]])
    M_aff = cv2.getAffineTransform(p_aff_src, p_aff_dst)
    warp_aff = cv2.warpAffine(rgb, M_aff, (w, h))

    # Perspective (4 points)
    p_persp_src = np.float32([[120, 80], [w - 120, 80], [40, h - 50], [w - 40, h - 50]])
    p_persp_dst = np.float32([[0, 0], [w, 0], [140, h], [w - 140, h]])
    M_persp = cv2.getPerspectiveTransform(p_persp_src, p_persp_dst)
    warp_persp = cv2.warpPerspective(rgb, M_persp, (w, h))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Input Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(warp_aff)
    axes[1].set_title("Affine Transformation (3 Points)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(warp_persp)
    axes[2].set_title("Perspective Transformation (4 Points)", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 60: Geometric Mapping Comparison (Affine vs Perspective)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp60] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demonstrate_transformations(os.path.join(script_dir, "input.jpg"))
