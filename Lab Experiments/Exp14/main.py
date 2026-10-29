"""
Experiment 14: Threshold-Based Image Segmentation
Objective: Convert source image to binary representation using different thresholding
values and techniques (manual threshold vs. Otsu's automatic optimal thresholding).
"""

import os
import cv2
import matplotlib.pyplot as plt

def threshold_segmentation(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Manual thresholds
    _, thresh_low = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    _, thresh_mid = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Otsu automatic optimal threshold
    otsu_val, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("Original Grayscale Input", fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(thresh_low, cmap='gray')
    axes[0, 1].set_title("Binary Threshold (T = 60)", fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(thresh_mid, cmap='gray')
    axes[1, 0].set_title("Binary Threshold (T = 127)", fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(thresh_otsu, cmap='gray')
    axes[1, 1].set_title(f"Otsu Optimal Threshold (T = {int(otsu_val)})", fontweight='bold')
    axes[1, 1].axis('off')

    plt.suptitle("Exp 14: Threshold-Based Image Segmentation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp14] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    threshold_segmentation(os.path.join(script_dir, "input.jpg"))
