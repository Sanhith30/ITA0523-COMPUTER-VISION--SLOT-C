"""
Experiment 33: RGB Channel Histogram Analysis
Objective: Separate color image into Blue, Green, Red channels, compute individual
intensity histograms, and interpret color distribution characteristics.
"""

import os
import cv2
import matplotlib.pyplot as plt

def analyze_rgb_histograms(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    b, g, r = cv2.split(bgr)

    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Input Color Image", fontweight='bold')
    axes[0, 0].axis('off')

    # Combined RGB Plot
    axes[0, 1].plot(hist_r, color='red', label='Red Channel', lw=1.8)
    axes[0, 1].plot(hist_g, color='green', label='Green Channel', lw=1.8)
    axes[0, 1].plot(hist_b, color='blue', label='Blue Channel', lw=1.8)
    axes[0, 1].set_title("Combined RGB Intensity Distributions", fontweight='bold')
    axes[0, 1].set_xlim([0, 256])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Individual Subplots
    axes[1, 0].plot(hist_r, color='crimson', lw=1.5, label='Red')
    axes[1, 0].plot(hist_g, color='forestgreen', lw=1.5, label='Green')
    axes[1, 0].set_title("Red & Green Channels")
    axes[1, 0].set_xlim([0, 256])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(hist_b, color='navy', lw=1.5, label='Blue')
    axes[1, 1].set_title("Blue Channel")
    axes[1, 1].set_xlim([0, 256])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Exp 33: Multi-Channel RGB Histogram Distribution Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp33] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    analyze_rgb_histograms(os.path.join(script_dir, "input.jpg"))
