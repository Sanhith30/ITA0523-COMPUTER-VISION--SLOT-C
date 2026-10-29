"""
Experiment 35: Grayscale Morphological Dilation and Structural Continuity
Objective: Apply dilation with custom structuring kernels, save the dilated image,
and analyze how dilation expands boundaries and bridges discontinuities.
"""

import os
import cv2
import matplotlib.pyplot as plt

def morphological_dilation_analysis(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(img, kernel, iterations=1)

    # Save output file
    save_path = os.path.join(os.path.dirname(image_path), "dilated_output.jpg")
    cv2.imwrite(save_path, dilated)

    diff = cv2.absdiff(dilated, img)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Input Grayscale Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(dilated, cmap='gray')
    axes[1].set_title("Dilated Image (7x7 Structuring Element)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title("Expanded Boundary Regions (|Diff|)", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 35: Morphological Dilation Boundary Expansion", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp35] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    morphological_dilation_analysis(os.path.join(script_dir, "input.jpg"))
