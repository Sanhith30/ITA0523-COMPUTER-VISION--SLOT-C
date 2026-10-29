"""
Experiment 65: ROI Selection and Relocation
Objective: Select a specific ROI from the image, copy its contents,
paste it into another image location, and verify that the target retains appearance.
"""

import os
import cv2
import matplotlib.pyplot as plt

def copy_paste_roi(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = bgr.shape[:2]
    annotated = bgr.copy()

    # Source ROI (Face region)
    sy1, sy2 = int(h * 0.12), int(h * 0.35)
    sx1, sx2 = int(w * 0.43), int(w * 0.55)
    roi = bgr[sy1:sy2, sx1:sx2]
    rh, rw = roi.shape[:2]

    # Destination (Top-Right area)
    dy1, dx1 = 30, w - rw - 30
    dy2, dx2 = dy1 + rh, dx1 + rw

    annotated[dy1:dy2, dx1:dx2] = roi

    # Annotate source and destination
    cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), (0, 255, 0), 3)
    cv2.putText(annotated, "Source ROI", (sx1, sy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), (0, 0, 255), 3)
    cv2.putText(annotated, "Pasted ROI", (dx1, dy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Save output file
    cv2.imwrite(os.path.join(os.path.dirname(image_path), "roi_relocated.jpg"), annotated)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Input Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Modified Image (Source Green, Relocated Red)", fontweight='bold')
    axes[1].axis('off')

    plt.suptitle("Exp 65: Image ROI Selection and Relocation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp65] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    copy_paste_roi(os.path.join(script_dir, "input.jpg"))
