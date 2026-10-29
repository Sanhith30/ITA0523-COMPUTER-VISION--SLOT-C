"""
Experiment 77: Morphological Erosion and Boundary Shrinkage
Objective: Apply morphological erosion to contract foreground objects,
save the eroded output, and comment on boundary loss and shrinkage.
"""

import os
import cv2
import matplotlib.pyplot as plt

def morphological_erosion_export(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    eroded = cv2.erode(img, kernel, iterations=1)

    save_path = os.path.join(os.path.dirname(image_path), "eroded_output.jpg")
    cv2.imwrite(save_path, eroded)

    diff = cv2.absdiff(img, eroded)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original Grayscale", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(eroded, cmap='gray')
    axes[1].set_title("Eroded Image (7x7 Kernel)\n(Boundary Contraction)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title("Eroded Boundary Loss (|Diff|)", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 77: Morphological Erosion & Structural Contraction", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp77] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    morphological_erosion_export(os.path.join(script_dir, "input.jpg"))
