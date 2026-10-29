"""
Experiment 26: Morphological Closing (Dilation followed by Erosion)
Objective: Apply morphological closing to bridge narrow gaps, fill small holes,
and improve contour connectivity and smoothness in foreground structures.
"""

import os
import cv2
import matplotlib.pyplot as plt

def morphological_closing(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Structuring elements
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    closing_k5 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k5)
    closing_k11 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k11)

    # Difference map showing closed regions (holes/discontinuities filled)
    diff = cv2.absdiff(closing_k11, img)

    fig, axes = plt.subplots(1, 4, figsize=(17, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original Grayscale Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(closing_k5, cmap='gray')
    axes[1].set_title("Closing (5x5 Rect Kernel)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(closing_k11, cmap='gray')
    axes[2].set_title("Closing (11x11 Ellipse Kernel)", fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(diff, cmap='hot')
    axes[3].set_title("Filled Gaps / Discontinuities (|Diff|)", fontweight='bold')
    axes[3].axis('off')

    plt.suptitle("Exp 26: Morphological Closing for Structural Continuity", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp26] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    morphological_closing(os.path.join(script_dir, "input.jpg"))
