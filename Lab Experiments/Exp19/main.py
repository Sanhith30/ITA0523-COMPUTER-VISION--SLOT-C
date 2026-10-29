"""
Experiment 19: Region of Interest (ROI) Manipulation
Objective: Select a specific rectangular ROI from the supplied image,
copy its contents, paste it into another location, and display the result.
"""

import os
import cv2
import matplotlib.pyplot as plt

def manipulate_roi(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = bgr.shape[:2]
    annotated = bgr.copy()

    # Source ROI (e.g. Messi's face region)
    sy1, sy2 = int(h * 0.12), int(h * 0.35)
    sx1, sx2 = int(w * 0.43), int(w * 0.55)
    roi = bgr[sy1:sy2, sx1:sx2]
    roi_h, roi_w = roi.shape[:2]

    # Destination coordinates (Top-Left corner area)
    dy1, dx1 = 30, 30
    dy2, dx2 = dy1 + roi_h, dx1 + roi_w

    # Paste ROI
    annotated[dy1:dy2, dx1:dx2] = roi

    # Draw bounding boxes for clarity
    cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), (0, 255, 0), 3) # Green = Source
    cv2.putText(annotated, "Source ROI", (sx1, sy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), (0, 0, 255), 3) # Red = Pasted Destination
    cv2.putText(annotated, "Pasted ROI", (dx1, dy1 + roi_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Input Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Modified Image (ROI Copied and Relocated)", fontweight='bold')
    axes[1].axis('off')

    plt.suptitle("Exp 19: Region of Interest (ROI) Selection and Duplication", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp19] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    manipulate_roi(os.path.join(script_dir, "input.jpg"))
