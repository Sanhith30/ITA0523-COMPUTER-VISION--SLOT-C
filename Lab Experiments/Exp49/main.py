"""
Experiment 49: Prominent Corner Detection with Harris Algorithm
Objective: Convert image to grayscale, compute response, isolate prominent corners,
mark detected points clearly, and save the annotated result.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_and_export_corners(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)

    # Sub-pixel corner refinement & visualization
    annotated = bgr.copy()
    threshold = 0.015 * dst.max()
    corner_coords = np.argwhere(dst > threshold)

    for pt in corner_coords:
        y, x = pt
        cv2.circle(annotated, (x, y), 3, (0, 255, 255), -1) # Yellow markers

    # Save output image
    save_path = os.path.join(os.path.dirname(image_path), "annotated_corners.jpg")
    cv2.imwrite(save_path, annotated)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Prominent Corners Annotated ({len(corner_coords)} Points)", fontweight='bold')
    axes[1].axis('off')

    plt.suptitle("Exp 49: Harris Corner Identification & Feature Export", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp49] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    detect_and_export_corners(os.path.join(script_dir, "input.jpg"))
