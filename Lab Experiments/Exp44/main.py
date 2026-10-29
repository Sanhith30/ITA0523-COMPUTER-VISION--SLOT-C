"""
Experiment 44: Directional Sobel Edge Filtering
Objective: Apply horizontal and vertical Sobel gradient filters, display both
directional responses, and fuse them to generate the overall edge map.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_filtering_directional(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    abs_x = cv2.convertScaleAbs(grad_x)
    abs_y = cv2.convertScaleAbs(grad_y)
    combined = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Input Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(abs_x, cmap='magma')
    axes[1].set_title("Horizontal Sobel (|Gx|)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(abs_y, cmap='magma')
    axes[2].set_title("Vertical Sobel (|Gy|)", fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(combined, cmap='magma')
    axes[3].set_title("Combined Directional Response", fontweight='bold')
    axes[3].axis('off')

    plt.suptitle("Exp 44: Horizontal and Vertical Sobel Edge Decomposition", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp44] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sobel_filtering_directional(os.path.join(script_dir, "input.jpg"))
