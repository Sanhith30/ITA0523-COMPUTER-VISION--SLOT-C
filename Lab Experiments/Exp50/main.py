"""
Experiment 50: Image Scaling to Half and Double Size
Objective: Scale a given image to half size (0.5x) and double size (2.0x) using
suitable interpolation methods (INTER_AREA for reduction, INTER_CUBIC for enlargement).
"""

import os
import cv2
import matplotlib.pyplot as plt

def scale_image_half_double(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # 1. Half Size (0.5x) using INTER_AREA
    half_size = cv2.resize(rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

    # 2. Double Size (2.0x) using INTER_CUBIC
    double_size = cv2.resize(rgb, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].imshow(rgb)
    axes[0].set_title(f"Source Image ({w}x{h})", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(half_size)
    axes[1].set_title(f"Half-Size (0.5x, {w//2}x{h//2})\n(Interpolation: INTER_AREA)", fontweight='bold')
    axes[1].axis('off')

    # Display center crop of double-size image to evaluate pixel sharpness
    dh, dw = double_size.shape[:2]
    crop_double = double_size[dh//4:3*dh//4, dw//4:3*dw//4]
    axes[2].imshow(crop_double)
    axes[2].set_title(f"Double-Size (2.0x, {w*2}x{h*2})\n(Interpolation: INTER_CUBIC - High Sharpness)", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 50: Multi-Scale Image Resizing (Half & Double Dimensions)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp50] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scale_image_half_double(os.path.join(script_dir, "input.jpg"))
