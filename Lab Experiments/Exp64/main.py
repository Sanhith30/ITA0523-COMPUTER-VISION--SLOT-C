"""
Experiment 64: Morphological Opening and Closing
Objective: Apply opening to suppress small isolated details and closing to bridge
narrow gaps and holes, analyzing resulting structural continuity.
"""

import os
import cv2
import matplotlib.pyplot as plt

def opening_closing_evaluation(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original Grayscale", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(opening, cmap='gray')
    axes[1].set_title("Morphological Opening\n(Erosion followed by Dilation)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(closing, cmap='gray')
    axes[2].set_title("Morphological Closing\n(Dilation followed by Erosion)", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 64: Morphological Opening vs Closing Operations", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp64] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    opening_closing_evaluation(os.path.join(script_dir, "input.jpg"))
