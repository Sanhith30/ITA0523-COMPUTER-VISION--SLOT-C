"""
Experiment 27: Background Separation Using Color-Based Thresholding
Objective: Generate a binary background mask by segmenting dark/neutral regions,
invert to obtain foreground, and display the isolated foreground output.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def separate_background(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Convert to Grayscale & HSV
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Dark background mask (intensity < 35)
    _, bg_mask = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY_INV)

    # Foreground mask (inverted background)
    fg_mask = cv2.bitwise_not(bg_mask)

    # Morphological clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask_clean = cv2.morphologyEx(fg_mask_clean, cv2.MORPH_CLOSE, kernel)

    # Apply mask
    foreground = cv2.bitwise_and(rgb, rgb, mask=fg_mask_clean)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Original Input Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(fg_mask_clean, cmap='gray')
    axes[1].set_title("Extracted Binary Foreground Mask", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(foreground)
    axes[2].set_title("Isolated Foreground Object (Background Removed)", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 27: Background Removal via Intensity/Color Thresholding", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp27] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    separate_background(os.path.join(script_dir, "input.jpg"))
