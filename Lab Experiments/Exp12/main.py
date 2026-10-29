"""
Experiment 12: Morphological Dilation on Grayscale Image
Objective: Apply morphological dilation using different structuring elements (kernel sizes)
to expand bright foreground boundaries and bridge gaps.
"""

import os
import cv2
import matplotlib.pyplot as plt

def apply_dilation(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Structuring elements
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    dilated_k3 = cv2.dilate(img, k3, iterations=1)
    dilated_k7 = cv2.dilate(img, k7, iterations=1)
    dilated_k11 = cv2.dilate(img, k11, iterations=1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    images = [
        ("Original Grayscale", img),
        ("Dilated (3x3 Rect)", dilated_k3),
        ("Dilated (7x7 Rect)", dilated_k7),
        ("Dilated (11x11 Ellipse)", dilated_k11)
    ]

    for ax, (title, pic) in zip(axes, images):
        ax.imshow(pic, cmap='gray')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.axis('off')

    plt.suptitle("Exp 12: Morphological Dilation with Varying Structuring Elements", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp12] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    apply_dilation(os.path.join(script_dir, "input.jpg"))
