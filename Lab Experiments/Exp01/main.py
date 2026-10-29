"""
Experiment 01: Gaussian Smoothing on Grayscale Image
Objective: Smooth a supplied grayscale image using Gaussian filtering with
user-selected kernel dimensions. Compare noise reduction and detail blurring.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_gaussian_smoothing(image_path, kernel_sizes=[3, 7, 15]):
    # Load input grayscale image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
        
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to decode image.")

    results = {"Original": img}
    for k in kernel_sizes:
        # Kernel size must be positive and odd
        if k % 2 == 0:
            k += 1
        smoothed = cv2.GaussianBlur(img, (k, k), sigmaX=0)
        results[f"Gaussian Blur ({k}x{k})"] = smoothed

    # Plot side-by-side comparison figure
    fig, axes = plt.subplots(1, len(results), figsize=(16, 4))
    for ax, (title, result_img) in zip(axes, results.items()):
        ax.imshow(result_img, cmap='gray')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle("Exp 01: Gaussian Smoothing with Varying Kernel Sizes", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp01] Successfully generated output screenshot at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_img = os.path.join(script_dir, "input.jpg")
    apply_gaussian_smoothing(input_img, kernel_sizes=[3, 9, 17])
