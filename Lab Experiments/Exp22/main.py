"""
Experiment 22: Image Resizing and Interpolation Method Comparison
Objective: Resize an image to enlarged (2x) and reduced (0.5x) dimensions using
Nearest Neighbor, Bilinear, Bicubic, and Area interpolation techniques.
"""

import os
import cv2
import matplotlib.pyplot as plt

def resize_interpolation_comparison(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Downscaling (0.3x) with INTER_AREA vs INTER_NEAREST
    small_area = cv2.resize(rgb, (int(w * 0.3), int(h * 0.3)), interpolation=cv2.INTER_AREA)
    small_near = cv2.resize(rgb, (int(w * 0.3), int(h * 0.3)), interpolation=cv2.INTER_NEAREST)

    # Upscaling small crop back up to evaluate interpolation artifacts
    crop = rgb[int(h*0.1):int(h*0.4), int(w*0.4):int(w*0.6)]
    crop_h, crop_w = crop.shape[:2]

    up_nearest = cv2.resize(crop, (crop_w * 3, crop_h * 3), interpolation=cv2.INTER_NEAREST)
    up_linear = cv2.resize(crop, (crop_w * 3, crop_h * 3), interpolation=cv2.INTER_LINEAR)
    up_cubic = cv2.resize(crop, (crop_w * 3, crop_h * 3), interpolation=cv2.INTER_CUBIC)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(rgb)
    axes[0].set_title(f"Original ({w}x{h})", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(up_nearest)
    axes[1].set_title("Upscaled: INTER_NEAREST\n(Notice Blocky Pixels)", fontweight='bold', fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(up_linear)
    axes[2].set_title("Upscaled: INTER_LINEAR\n(Standard Bilinear)", fontweight='bold', fontsize=10)
    axes[2].axis('off')

    axes[3].imshow(up_cubic)
    axes[3].set_title("Upscaled: INTER_CUBIC\n(Smooth & Sharp)", fontweight='bold', fontsize=10)
    axes[3].axis('off')

    plt.suptitle("Exp 22: Image Resizing and Interpolation Method Quality Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp22] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resize_interpolation_comparison(os.path.join(script_dir, "input.jpg"))
