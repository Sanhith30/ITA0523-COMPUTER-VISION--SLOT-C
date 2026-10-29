"""
Experiment 09: Grayscale Image Enhancement using Histogram Equalization
Objective: Enhance contrast, display original and equalized images,
and compute Cumulative Distribution Function (CDF) and histograms.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_histogram_enhancement(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Equalization
    enhanced = cv2.equalizeHist(img)

    # Compute CDF for original and enhanced
    hist_orig, _ = np.histogram(img.flatten(), 256, [0, 256])
    cdf_orig = hist_orig.cumsum()
    cdf_orig_norm = cdf_orig * float(hist_orig.max()) / cdf_orig.max()

    hist_enh, _ = np.histogram(enhanced.flatten(), 256, [0, 256])
    cdf_enh = hist_enh.cumsum()
    cdf_enh_norm = cdf_enh * float(hist_enh.max()) / cdf_enh.max()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title("Input Image (Underexposed/Low Contrast)", fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(enhanced, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title("Equalized Enhanced Image", fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].plot(cdf_orig_norm, color='blue', label='CDF', lw=2)
    axes[1, 0].hist(img.flatten(), 256, [0, 256], color='gray', alpha=0.6, label='Histogram')
    axes[1, 0].set_title("Original Intensity Distribution & CDF")
    axes[1, 0].legend(loc='upper left')
    axes[1, 0].set_xlim([0, 256])
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(cdf_enh_norm, color='blue', label='Linear CDF', lw=2)
    axes[1, 1].hist(enhanced.flatten(), 256, [0, 256], color='navy', alpha=0.6, label='Histogram')
    axes[1, 1].set_title("Equalized Intensity Distribution & Linear CDF")
    axes[1, 1].legend(loc='upper left')
    axes[1, 1].set_xlim([0, 256])
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Exp 09: Detailed Grayscale Histogram Equalization Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp09] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    analyze_histogram_enhancement(os.path.join(script_dir, "input.jpg"))
