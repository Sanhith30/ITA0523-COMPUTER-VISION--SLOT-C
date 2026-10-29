"""
Experiment 41: Canny Edge Detection with Threshold Parameter Tuning
Objective: Convert input image to grayscale, apply Canny edge detector with
lower and upper threshold values, and analyze boundary and structural detection.
"""

import os
import cv2
import matplotlib.pyplot as plt

def canny_detector(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges_50_150 = cv2.Canny(gray, 50, 150)
    edges_100_200 = cv2.Canny(gray, 100, 200)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Source Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(edges_50_150, cmap='gray')
    axes[1].set_title("Canny Edges (Thresholds: 50, 150)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(edges_100_200, cmap='gray')
    axes[2].set_title("Canny Edges (Thresholds: 100, 200)", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 41: Canny Edge Detection & Boundary Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp41] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    canny_detector(os.path.join(script_dir, "input.jpg"))
