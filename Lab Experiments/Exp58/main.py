"""
Experiment 58: Clockwise Image Rotations (90, 180, 270 Degrees)
Objective: Apply OpenCV rotations, handle dimensions automatically,
and present visual orientation comparisons.
"""

import os
import cv2
import matplotlib.pyplot as plt

def rotate_three_orientations(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rot90 = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
    rot180 = cv2.rotate(rgb, cv2.ROTATE_180)
    rot270 = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    images = [
        (f"Original ({rgb.shape[1]}x{rgb.shape[0]})", rgb),
        (f"90° CW ({rot90.shape[1]}x{rot90.shape[0]})", rot90),
        (f"180° CW ({rot180.shape[1]}x{rot180.shape[0]})", rot180),
        (f"270° CW ({rot270.shape[1]}x{rot270.shape[0]})", rot270)
    ]

    for ax, (title, pic) in zip(axes, images):
        ax.imshow(pic)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.axis('off')

    plt.suptitle("Exp 58: Rotational Transforms Across 90°, 180°, and 270°", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp58] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rotate_three_orientations(os.path.join(script_dir, "input.jpg"))
