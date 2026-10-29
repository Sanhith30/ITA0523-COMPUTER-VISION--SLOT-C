"""
Experiment 02: Histogram Equalization for Grayscale Contrast Enhancement
Objective: Improve contrast of a grayscale image using histogram equalization.
Display input and enhanced images alongside their intensity histograms.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Apply Histogram Equalization
    equalized = cv2.equalizeHist(img)

    # Compute Histograms
    hist_orig = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256])

    # Plot 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Images
    axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title("Input Image (Low Contrast)", fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(equalized, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title("Equalized Image (Enhanced Contrast)", fontweight='bold')
    axes[0, 1].axis('off')

    # Histograms
    axes[1, 0].plot(hist_orig, color='black', lw=1.5)
    axes[1, 0].set_title("Original Intensity Histogram")
    axes[1, 0].set_xlim([0, 256])
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(hist_eq, color='navy', lw=1.5)
    axes[1, 1].set_title("Equalized Intensity Histogram (Uniform Spread)")
    axes[1, 1].set_xlim([0, 256])
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Exp 02: Contrast Improvement via Histogram Equalization", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp02] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    histogram_equalization(os.path.join(script_dir, "input.jpg"))
