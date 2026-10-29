"""
Experiment 25: Harris Corner Detection
Objective: Convert supplied image to grayscale, compute Harris Corner Response R,
and highlight prominent corners using thresholding and non-maximum dilation.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def harris_corner_detection(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)

    # Harris Corner Detector: blockSize=2, ksize=3, k=0.04
    dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)

    # Dilate corner points to mark them clearly
    dst_dilated = cv2.dilate(dst, None)

    # Highlight corners in red on original image
    annotated = bgr.copy()
    threshold = 0.01 * dst.max()
    annotated[dst_dilated > threshold] = [0, 0, 255] # BGR Red

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(dst, cmap='hot')
    axes[1].set_title("Harris Corner Response Map (R)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Detected Corners Highlighted (Red Markers)", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 25: Harris Corner Detection & Feature Localization", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp25] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    harris_corner_detection(os.path.join(script_dir, "input.jpg"))
