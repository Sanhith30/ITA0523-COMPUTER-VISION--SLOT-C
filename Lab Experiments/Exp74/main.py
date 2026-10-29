"""
Experiment 74: Image Resizing Across Multiple Interpolation Techniques
Objective: Resize image to smaller and larger dimensions, evaluating sharpness,
pixelation, and visual quality across Nearest Neighbor, Linear, and Cubic algorithms.
"""

import os
import cv2
import matplotlib.pyplot as plt

def evaluate_interpolation_methods(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Crop interesting region (Face)
    crop = rgb[int(h*0.12):int(h*0.35), int(w*0.43):int(w*0.55)]
    ch, cw = crop.shape[:2]

    up_nearest = cv2.resize(crop, (cw * 4, ch * 4), interpolation=cv2.INTER_NEAREST)
    up_linear = cv2.resize(crop, (cw * 4, ch * 4), interpolation=cv2.INTER_LINEAR)
    up_cubic = cv2.resize(crop, (cw * 4, ch * 4), interpolation=cv2.INTER_CUBIC)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(crop)
    axes[0].set_title(f"Target Crop ({cw}x{ch})", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(up_nearest)
    axes[1].set_title("4x: INTER_NEAREST\n(Pixelated/Staircasing)", fontweight='bold', fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(up_linear)
    axes[2].set_title("4x: INTER_LINEAR\n(Bilinear Smoothing)", fontweight='bold', fontsize=10)
    axes[2].axis('off')

    axes[3].imshow(up_cubic)
    axes[3].set_title("4x: INTER_CUBIC\n(High Fidelity / Sharp)", fontweight='bold', fontsize=10)
    axes[3].axis('off')

    plt.suptitle("Exp 74: Scaling Interpolation Quality Evaluation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp74] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    evaluate_interpolation_methods(os.path.join(script_dir, "input.jpg"))
