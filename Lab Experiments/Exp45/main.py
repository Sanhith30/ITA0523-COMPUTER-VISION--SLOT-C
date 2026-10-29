"""
Experiment 45: Sequential Morphological Erosion and Dilation
Objective: Apply morphological erosion to contract foreground, then apply dilation
to expand foreground, and discuss structural modifications.
"""

import os
import cv2
import matplotlib.pyplot as plt

def erosion_then_dilation(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    eroded = cv2.erode(img, kernel, iterations=1)
    dilated_from_eroded = cv2.dilate(eroded, kernel, iterations=1)
    pure_dilated = cv2.dilate(img, kernel, iterations=1)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    images = [
        ("Original Grayscale", img),
        ("Eroded Image\n(Shrunk Boundaries)", eroded),
        ("Dilated from Eroded\n(Opening Process)", dilated_from_eroded),
        ("Pure Dilated Image\n(Expanded Boundaries)", pure_dilated)
    ]

    for ax, (title, pic) in zip(axes, images):
        ax.imshow(pic, cmap='gray')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.axis('off')

    plt.suptitle("Exp 45: Morphological Erosion vs Dilation & Sequential Effects", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp45] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    erosion_then_dilation(os.path.join(script_dir, "input.jpg"))
