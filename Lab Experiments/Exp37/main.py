"""
Experiment 37: Sobel Directional Filtering & Magnitude Synthesis
Objective: Apply horizontal and vertical Sobel filters to identify directional edges,
and combine both gradients into a comprehensive magnitude edge map.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_magnitude_analysis(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    abs_gx = cv2.convertScaleAbs(gx)
    abs_gy = cv2.convertScaleAbs(gy)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Input Grayscale", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(abs_gx, cmap='magma')
    axes[1].set_title("Sobel Gx (Vertical Transitions)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(abs_gy, cmap='magma')
    axes[2].set_title("Sobel Gy (Horizontal Transitions)", fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(mag, cmap='magma')
    axes[3].set_title("Combined Gradient Magnitude", fontweight='bold')
    axes[3].axis('off')

    plt.suptitle("Exp 37: Directional Sobel Derivatives and Combined Gradient", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp37] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sobel_magnitude_analysis(os.path.join(script_dir, "input.jpg"))
