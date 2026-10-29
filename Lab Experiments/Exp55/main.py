"""
Experiment 55: Morphological Erosion and Dilation Side-by-Side
Objective: Implement both erosion and dilation on a grayscale image,
compare their influence on boundaries, gaps, and foreground shapes.
"""

import os
import cv2
import matplotlib.pyplot as plt

def erosion_and_dilation_comparison(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    eroded = cv2.erode(img, kernel, iterations=1)
    dilated = cv2.dilate(img, kernel, iterations=1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original Grayscale Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(eroded, cmap='gray')
    axes[1].set_title("Eroded Image\n(Contracts Objects, Shrinks Boundaries)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(dilated, cmap='gray')
    axes[2].set_title("Dilated Image\n(Expands Objects, Fills Inward Gaps)", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 55: Morphological Erosion vs Dilation Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp55] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    erosion_and_dilation_comparison(os.path.join(script_dir, "input.jpg"))
