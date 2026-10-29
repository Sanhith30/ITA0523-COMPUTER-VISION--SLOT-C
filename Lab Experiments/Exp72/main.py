"""
Experiment 72: Exact 180-Degree Clockwise Rotation
Objective: Rotate supplied image exactly 180 degrees clockwise using cv2.rotate,
verify that dimensions remain consistent with source, and save output file.
"""

import os
import cv2
import matplotlib.pyplot as plt

def rotate_180_degrees(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    # 180 degree rotation
    rot180 = cv2.rotate(bgr, cv2.ROTATE_180)

    save_path = os.path.join(os.path.dirname(image_path), "rotated_180.jpg")
    cv2.imwrite(save_path, rot180)

    h, w = bgr.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original Input Image ({w}x{h})", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(rot180, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"180° Inverted Output ({w}x{h})", fontweight='bold')
    axes[1].axis('off')

    plt.suptitle("Exp 72: Exact 180-Degree Image Inversion", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp72] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rotate_180_degrees(os.path.join(script_dir, "input.jpg"))
