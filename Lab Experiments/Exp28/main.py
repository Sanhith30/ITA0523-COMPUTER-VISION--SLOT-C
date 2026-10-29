"""
Experiment 28: Image Rotation Exactly 270 Degrees Clockwise
Objective: Rotate supplied image exactly 270 degrees clockwise (equivalent to 90 degrees CCW),
ensure dimensions are handled accurately, save transformed file, and display comparison.
"""

import os
import cv2
import matplotlib.pyplot as plt

def rotate_270_clockwise(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    # 270 CW = cv2.ROTATE_90_COUNTERCLOCKWISE
    rotated_270 = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Save output rotated file
    save_path = os.path.join(os.path.dirname(image_path), "rotated_270.jpg")
    cv2.imwrite(save_path, rotated_270)

    orig_h, orig_w = bgr.shape[:2]
    rot_h, rot_w = rotated_270.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original Image ({orig_w}x{orig_h})", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(rotated_270, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"270° Clockwise Rotated ({rot_w}x{rot_h})", fontweight='bold')
    axes[1].axis('off')

    plt.suptitle("Exp 28: Exact 270-Degree Clockwise Image Rotation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp28] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rotate_270_clockwise(os.path.join(script_dir, "input.jpg"))
