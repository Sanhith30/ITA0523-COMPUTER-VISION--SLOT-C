"""
Experiment 04: Sobel-Based Edge Detection
Objective: Apply horizontal and vertical Sobel gradient operators to an input image.
Display directional gradient images independently, then combine them to generate an overall edge map.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_edge_detection(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Compute Sobel Gradients (CV_64F avoids negative clipping)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) # Vertical edges (dx)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) # Horizontal edges (dy)

    # Absolute conversion to 8-bit
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    # Combined gradient magnitude: sqrt(Gx^2 + Gy^2)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title("Input Grayscale Image", fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(abs_sobel_x, cmap='gray')
    axes[0, 1].set_title("Sobel Horizontal Derivative (Vertical Boundaries)", fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(abs_sobel_y, cmap='gray')
    axes[1, 0].set_title("Sobel Vertical Derivative (Horizontal Boundaries)", fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(magnitude, cmap='gray')
    axes[1, 1].set_title("Combined Sobel Gradient Magnitude", fontweight='bold')
    axes[1, 1].axis('off')

    plt.suptitle("Exp 04: Directional Sobel Gradients & Overall Edge Map", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp04] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sobel_edge_detection(os.path.join(script_dir, "input.jpg"))
