"""
Experiment 17: Canny Edge Detection
Objective: Convert input image to grayscale, select appropriate lower and upper
hysteresis thresholds, and evaluate edge maps across threshold settings.
"""

import os
import cv2
import matplotlib.pyplot as plt

def canny_edge_detection(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny with 3 different threshold configurations
    canny_low = cv2.Canny(gray, threshold1=30, threshold2=90)
    canny_mid = cv2.Canny(gray, threshold1=70, threshold2=180)
    canny_high = cv2.Canny(gray, threshold1=120, threshold2=250)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    images = [
        ("Original Image", cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None),
        ("Canny (30, 90) - Dense Edges", canny_low, 'gray'),
        ("Canny (70, 180) - Optimal Edges", canny_mid, 'gray'),
        ("Canny (120, 250) - Strong Edges", canny_high, 'gray')
    ]

    for ax, (title, pic, cmap) in zip(axes, images):
        if cmap:
            ax.imshow(pic, cmap=cmap)
        else:
            ax.imshow(pic)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.axis('off')

    plt.suptitle("Exp 17: Canny Edge Detection & Hysteresis Threshold Evaluation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp17] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    canny_edge_detection(os.path.join(script_dir, "input.jpg"))
