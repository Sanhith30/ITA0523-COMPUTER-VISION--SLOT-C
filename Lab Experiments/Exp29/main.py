"""
Experiment 29: Color to Grayscale Representation Conversion
Objective: Read supplied color image, convert it into grayscale using cv2.cvtColor,
display both representations separately, and save the converted file.
"""

import os
import cv2
import matplotlib.pyplot as plt

def convert_to_grayscale(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert BGR to Grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Save converted file
    save_path = os.path.join(os.path.dirname(image_path), "grayscale_output.jpg")
    cv2.imwrite(save_path, gray)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original BGR Color Image ({bgr.shape[2]} Channels)", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title("Converted Grayscale Image (1 Channel)", fontweight='bold')
    axes[1].axis('off')

    plt.suptitle("Exp 29: BGR to Grayscale Image Conversion & Export", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp29] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    convert_to_grayscale(os.path.join(script_dir, "input.jpg"))
