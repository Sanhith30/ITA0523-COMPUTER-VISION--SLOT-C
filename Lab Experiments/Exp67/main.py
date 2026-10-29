"""
Experiment 67: Gaussian Smoothing and Noise Suppression
Objective: Apply Gaussian blur with varying kernel sizes (5x5, 13x13, 25x25),
and observe how kernel dimension influences edge softness and detail retention.
"""

import os
import cv2
import matplotlib.pyplot as plt

def evaluate_gaussian_kernels(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k5 = cv2.GaussianBlur(gray, (5, 5), 0)
    k13 = cv2.GaussianBlur(gray, (13, 13), 0)
    k25 = cv2.GaussianBlur(gray, (25, 25), 0)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    images = [
        ("Original Grayscale", gray),
        ("Gaussian (5x5 Kernel)\n(Subtle Smoothing)", k5),
        ("Gaussian (13x13 Kernel)\n(Moderate Noise Suppression)", k13),
        ("Gaussian (25x25 Kernel)\n(Heavy Structural Blurring)", k25)
    ]

    for ax, (title, pic) in zip(axes, images):
        ax.imshow(pic, cmap='gray')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.axis('off')

    plt.suptitle("Exp 67: Gaussian Filtering Kernel Parameter Exploration", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp67] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    evaluate_gaussian_kernels(os.path.join(script_dir, "input.jpg"))
