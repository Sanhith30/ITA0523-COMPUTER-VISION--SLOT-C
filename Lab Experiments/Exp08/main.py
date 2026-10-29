"""
Experiment 08: Foreground Object Extraction Using Color-Based Thresholding
Objective: Define color thresholds in HSV space, generate a binary mask,
and isolate the foreground object from the background.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_based_segmentation(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Convert to HSV color space
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Red color range wraps around 0/180 in OpenCV HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological cleaning to remove isolated noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # Apply mask to extract foreground
    extracted_bgr = cv2.bitwise_and(bgr, bgr, mask=mask_clean)
    extracted_rgb = cv2.cvtColor(extracted_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Input Color Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(mask_clean, cmap='gray')
    axes[1].set_title("Binary Segmentation Mask", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(extracted_rgb)
    axes[2].set_title("Extracted Foreground Object", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 08: Color-Based Foreground Segmentation (HSV Thresholding)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp08] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    color_based_segmentation(os.path.join(script_dir, "input.jpg"))
