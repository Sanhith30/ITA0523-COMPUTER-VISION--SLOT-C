"""
Experiment 75: Foreground Extraction Using Color-Level Thresholding
Objective: Select intensity thresholds, generate binary mask, isolate foreground,
and compare extracted subject against original image.
"""

import os
import cv2
import matplotlib.pyplot as plt

def extract_foreground_subject(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    fg = cv2.bitwise_and(rgb, rgb, mask=mask_clean)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Original Input Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(mask_clean, cmap='gray')
    axes[1].set_title("Binary Extraction Mask", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(fg)
    axes[2].set_title("Extracted Foreground Object", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 75: Foreground Extraction via Binary Masking", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp75] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extract_foreground_subject(os.path.join(script_dir, "input.jpg"))
