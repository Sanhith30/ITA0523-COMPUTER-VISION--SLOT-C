"""
Experiment 79: RGB Intensity Distribution Analysis
Objective: Separate image into R, G, B channels, generate individual histograms,
and interpret channel distributions in relation to overall color appearance.
"""

import os
import cv2
import matplotlib.pyplot as plt

def analyze_channel_intensities(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    b, g, r = cv2.split(bgr)

    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title("Input Color Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].plot(hist_r, color='crimson', lw=1.5)
    axes[1].set_title("Red Channel Intensity", fontweight='bold')
    axes[1].set_xlim([0, 256])
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(hist_g, color='forestgreen', lw=1.5)
    axes[2].set_title("Green Channel Intensity", fontweight='bold')
    axes[2].set_xlim([0, 256])
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(hist_b, color='navy', lw=1.5)
    axes[3].set_title("Blue Channel Intensity", fontweight='bold')
    axes[3].axis('off')
    axes[3].axis('on')
    axes[3].set_xlim([0, 256])
    axes[3].grid(True, alpha=0.3)

    plt.suptitle("Exp 79: Multi-Channel RGB Spectral Intensity Distribution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp79] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    analyze_channel_intensities(os.path.join(script_dir, "input.jpg"))
