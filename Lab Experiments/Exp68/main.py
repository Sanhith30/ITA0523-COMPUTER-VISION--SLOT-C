"""
Experiment 68: Top-Hat and Black-Hat Morphological Transformations
Objective: Apply Top-Hat to isolate bright details on dark background,
and Black-Hat to isolate dark details on bright background.
"""

import os
import cv2
import matplotlib.pyplot as plt

def tophat_blackhat_transforms(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Input Grayscale", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(tophat, cmap='hot')
    axes[1].set_title("Top-Hat Filter (15x15 Kernel)\n(Bright Details on Dark Background)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(blackhat, cmap='hot')
    axes[2].set_title("Black-Hat Filter (15x15 Kernel)\n(Dark Details on Bright Background)", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 68: Dual Morphological Extraction (Top-Hat & Black-Hat)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp68] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tophat_blackhat_transforms(os.path.join(script_dir, "input.jpg"))
