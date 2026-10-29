"""
Experiment 23: Object Recognition - Watch Detection and Localization
Objective: Implement template matching and feature recognition to locate a watch
within the supplied image, draw bounding boxes, and evaluate confidence.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_watch(image_path, template_path):
    img = cv2.imread(image_path)
    template = cv2.imread(template_path)
    if img is None or template is None:
        raise ValueError("Could not load image or template.")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    templ_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    th, tw = templ_gray.shape[:2]

    # Template Matching using Normalized Cross-Correlation
    res = cv2.matchTemplate(img_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + tw, top_left[1] + th)

    annotated = img.copy()
    cv2.rectangle(annotated, top_left, bottom_right, (0, 255, 0), 3)
    cv2.putText(annotated, f"Detected Watch ({max_val*100:.1f}%)", (top_left[0] - 20, top_left[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Target Object Template\n({tw}x{th})", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(res, cmap='hot')
    axes[1].set_title(f"Correlation Response Map\n(Peak Confidence: {max_val:.3f})", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Localized Watch Object (Bounding Box)", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Exp 23: Object Recognition (Watch Localization via Template Matching)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp23] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    detect_watch(
        os.path.join(script_dir, "input.jpg"),
        os.path.join(script_dir, "watch_template.png")
    )
