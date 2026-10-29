"""
Experiment 57: Grayscale Histogram Equalization & Brightness Redistribution
Objective: Enhance contrast through histogram equalization, generate histograms,
and observe how equalization reveals details in dark or light regions.
"""

import os
import cv2
import matplotlib.pyplot as plt

def equalize_histogram_details(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    equalized = cv2.equalizeHist(img)

    hist_orig = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("Input Image", fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(equalized, cmap='gray')
    axes[0, 1].set_title("Equalized Image", fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].plot(hist_orig, color='black')
    axes[1, 0].set_title("Input Histogram Distribution")
    axes[1, 0].set_xlim([0, 256])
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(hist_eq, color='navy')
    axes[1, 1].set_title("Equalized Uniform Distribution")
    axes[1, 1].set_xlim([0, 256])
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Exp 57: Histogram Equalization & Brightness Redistribution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp57] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    equalize_histogram_details(os.path.join(script_dir, "input.jpg"))
