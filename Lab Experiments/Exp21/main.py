"""
Experiment 21: Morphological Erosion on Grayscale Image
Objective: Apply morphological erosion with different structuring elements
to contract foreground boundaries and eliminate small isolated bright details.
"""

import os
import cv2
import matplotlib.pyplot as plt

def apply_erosion(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Structuring elements
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    eroded_k3 = cv2.erode(img, k3, iterations=1)
    eroded_k7 = cv2.erode(img, k7, iterations=1)
    eroded_k11 = cv2.erode(img, k11, iterations=1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    images = [
        ("Original Grayscale", img),
        ("Eroded (3x3 Rect)", eroded_k3),
        ("Eroded (7x7 Rect)", eroded_k7),
        ("Eroded (11x11 Ellipse)", eroded_k11)
    ]

    for ax, (title, pic) in zip(axes, images):
        ax.imshow(pic, cmap='gray')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.axis('off')

    plt.suptitle("Exp 21: Morphological Erosion with Multiple Structuring Elements", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp21] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    apply_erosion(os.path.join(script_dir, "input.jpg"))
