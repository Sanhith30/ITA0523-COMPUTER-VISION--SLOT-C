"""
Experiment 32: Top-Hat Morphological Transformation
Objective: Apply Top-Hat transformation (Difference between input and its morphological opening)
to extract and isolate small bright elements on dark or uneven backgrounds.
"""

import os
import cv2
import matplotlib.pyplot as plt

def tophat_transformation(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Structuring elements of different sizes
    k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    k21 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    tophat_k9 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k9)
    tophat_k21 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k21)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Input Grayscale Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(tophat_k9, cmap='hot')
    axes[1].set_title("Top-Hat Filter (9x9 Kernel)\n(Highlights Fine Bright Details)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(tophat_k21, cmap='hot')
    axes[2].set_title("Top-Hat Filter (21x21 Kernel)\n(Highlights Larger Bright Features)", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 32: Top-Hat Morphological Transformation (Bright Feature Isolation)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp32] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tophat_transformation(os.path.join(script_dir, "input.jpg"))
