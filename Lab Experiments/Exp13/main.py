"""
Experiment 13: Gaussian Blur Smoothing
Objective: Convert input image to grayscale if necessary, apply Gaussian filtering
with user-specified kernel dimensions, and compare noise reduction and structural preservation.
"""

import os
import cv2
import matplotlib.pyplot as plt

def gaussian_smoothing(image_path, kernel_size=9):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigmaX=0)
    
    # Difference map to highlight removed fine details/noise
    diff = cv2.absdiff(gray, blurred)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title("Original Grayscale Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(blurred, cmap='gray')
    axes[1].set_title(f"Gaussian Filtered (Kernel: {kernel_size}x{kernel_size})", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title("High-Frequency Details Removed (|Diff|)", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 13: Gaussian Blur & Structural Detail Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp13] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gaussian_smoothing(os.path.join(script_dir, "input.jpg"), kernel_size=11)
