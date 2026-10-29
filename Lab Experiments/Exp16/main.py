"""
Experiment 16: Visible Text Watermark on Image
Objective: Add a readable, styled, semi-transparent text watermark to a supplied image,
display the result, and save it as a new file while preserving the original.
"""

import os
import cv2
import matplotlib.pyplot as plt

def add_text_watermark(image_path, watermark_text="ITA05 CV LAB © 2026"):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = bgr.shape[:2]
    overlay = bgr.copy()

    # Watermark configuration
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 3
    color = (255, 255, 255) # White watermark

    # Position in bottom-right corner with padding
    (text_w, text_h), baseline = cv2.getTextSize(watermark_text, font, font_scale, font_thickness)
    x = w - text_w - 30
    y = h - 30

    # Draw semi-transparent background banner
    cv2.rectangle(overlay, (x - 10, y - text_h - 10), (x + text_w + 10, y + baseline + 10), (20, 20, 20), -1)
    # Draw text
    cv2.putText(overlay, watermark_text, (x, y), font, font_scale, color, font_thickness, cv2.LINE_AA)

    # Alpha blending: 70% original + 30% overlay banner/text
    alpha = 0.65
    watermarked = cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)

    # Save output watermarked image
    save_path = os.path.join(os.path.dirname(image_path), "watermarked_image.jpg")
    cv2.imwrite(save_path, watermarked)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Input Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Watermarked Image (Semi-Transparent Overlay)", fontweight='bold')
    axes[1].axis('off')

    plt.suptitle("Exp 16: Visible Text Watermarking with OpenCV", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp16] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    add_text_watermark(os.path.join(script_dir, "input.jpg"))
