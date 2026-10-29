"""
Experiment 61: Visible Text Watermarking on Image
Objective: Add a clear, styled watermark to the supplied image,
display the watermarked result, and save it while preserving the source.
"""

import os
import cv2
import matplotlib.pyplot as plt

def apply_watermark(image_path, text="OFFICIAL CV LAB 2026"):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = bgr.shape[:2]
    overlay = bgr.copy()

    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.0
    thickness = 2
    color = (255, 255, 255)

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = w - tw - 40
    y = h - 40

    # Shaded backing pill
    cv2.rectangle(overlay, (x - 12, y - th - 12), (x + tw + 12, y + baseline + 12), (10, 10, 10), -1)
    cv2.putText(overlay, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    # 60% overlay + 40% original
    watermarked = cv2.addWeighted(overlay, 0.65, bgr, 0.35, 0)

    save_path = os.path.join(os.path.dirname(image_path), "watermarked_output.jpg")
    cv2.imwrite(save_path, watermarked)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Input Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Watermarked Image (Clean Visible Overlay)", fontweight='bold')
    axes[1].axis('off')

    plt.suptitle("Exp 61: Image Watermarking & Preservation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp61] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    apply_watermark(os.path.join(script_dir, "input.jpg"))
