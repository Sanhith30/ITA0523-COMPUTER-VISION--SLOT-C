"""
Experiment 03: Image Rotation (90, 180, and 270 Degrees Clockwise)
Objective: Rotate supplied image clockwise through 90, 180, and 270 degrees,
maintaining proper dimensions and aspect proportions.
"""

import os
import cv2
import matplotlib.pyplot as plt

def rotate_image_series(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Apply rotations
    rot90 = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
    rot180 = cv2.rotate(rgb, cv2.ROTATE_180)
    rot270 = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE) # 270 CW == 90 CCW

    images = [
        (f"Original ({rgb.shape[1]}x{rgb.shape[0]})", rgb),
        (f"90 deg Clockwise ({rot90.shape[1]}x{rot90.shape[0]})", rot90),
        (f"180 deg Clockwise ({rot180.shape[1]}x{rot180.shape[0]})", rot180),
        (f"270 deg Clockwise ({rot270.shape[1]}x{rot270.shape[0]})", rot270)
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, (title, img) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.suptitle("Exp 03: Image Rotation through 90, 180, and 270 Degrees", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp03] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rotate_image_series(os.path.join(script_dir, "input.jpg"))
