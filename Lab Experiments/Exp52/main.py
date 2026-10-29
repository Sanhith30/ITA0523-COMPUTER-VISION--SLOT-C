"""
Experiment 52: Threshold-Based Segmentation Analysis
Objective: Convert source image to grayscale, select thresholding techniques
(Global Binary, Otsu, Adaptive Gaussian), and evaluate segmentation effectiveness.
"""

import os
import cv2
import matplotlib.pyplot as plt

def threshold_analysis(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    _, thresh_global = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    otsu_val, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 15, 2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("Input Grayscale Image", fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(thresh_global, cmap='gray')
    axes[0, 1].set_title("Global Binary Threshold (T = 127)", fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(thresh_otsu, cmap='gray')
    axes[1, 0].set_title(f"Otsu Optimal Threshold (T = {int(otsu_val)})", fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(thresh_adapt, cmap='gray')
    axes[1, 1].set_title("Adaptive Gaussian Thresholding", fontweight='bold')
    axes[1, 1].axis('off')

    plt.suptitle("Exp 52: Comparative Threshold-Based Segmentation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp52] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    threshold_analysis(os.path.join(script_dir, "input.jpg"))
