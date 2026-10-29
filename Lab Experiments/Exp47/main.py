"""
Experiment 47: Harris Corner Detection & Parameter Optimization
Objective: Calculate Harris Corner Response, adjust detection threshold
to eliminate false positives, and mark robust corners on image.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def harris_tuning(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    dst_dilated = cv2.dilate(dst, None)

    # Low threshold vs Optimal threshold
    annot_low = bgr.copy()
    annot_opt = bgr.copy()

    annot_low[dst_dilated > 0.005 * dst.max()] = [0, 0, 255] # Many false corners
    annot_opt[dst_dilated > 0.02 * dst.max()] = [0, 255, 0]   # Precise corners

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(annot_low, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Low Threshold (0.5% max)\n(High Sensitivity, More False Positives)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(annot_opt, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Optimal Threshold (2.0% max)\n(Robust Corner Salience)", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 47: Harris Corner Sensitivity & Threshold Tuning", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp47] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    harris_tuning(os.path.join(script_dir, "input.jpg"))
