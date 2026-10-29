"""
Experiment 73: Canny Edge Detection and Contour Mapping
Objective: Convert input to grayscale, select appropriate thresholds,
generate edge map, and analyze object boundaries, contours, and transitions.
"""

import os
import cv2
import matplotlib.pyplot as plt

def evaluate_canny_edges(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny_map = cv2.Canny(gray, 60, 160)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Input Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(canny_map, cmap='gray')
    axes[1].set_title("Canny Structural Edge Map (Thresholds: 60, 160)", fontweight='bold')
    axes[1].axis('off')

    plt.suptitle("Exp 73: Canny Boundary Extraction & Contour Mapping", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp73] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    evaluate_canny_edges(os.path.join(script_dir, "input.jpg"))
