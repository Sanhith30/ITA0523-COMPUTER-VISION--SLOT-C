"""
Experiment 05: Sobel Operator Gradient Combination and Analysis
Objective: Apply separate horizontal and vertical filters, calculate their combined
response, and analyze how directional gradients contribute to complete boundary detection.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_sobel_combination(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Gradient calculations
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # Linear combination using addWeighted
    combined_weighted = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    # L2 magnitude
    combined_l2 = cv2.magnitude(grad_x, grad_y)
    combined_l2 = np.clip(combined_l2, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Source Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(abs_grad_x, cmap='magma')
    axes[1].set_title("Sobel X (|dI/dx|)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(abs_grad_y, cmap='magma')
    axes[2].set_title("Sobel Y (|dI/dy|)", fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(combined_l2, cmap='magma')
    axes[3].set_title("Combined Gradient L2 Map", fontweight='bold')
    axes[3].axis('off')

    plt.suptitle("Exp 05: Edge Detection with Sobel Directional Operators & Fusion", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp05] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    analyze_sobel_combination(os.path.join(script_dir, "input.jpg"))
