"""
Experiment 76: White Image Canvas with Geometric Shapes & Text
Objective: Create a white image, draw a rectangle and a circle, add user-defined text
with suitable font properties, and save to a file.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def render_shapes_canvas(width=720, height=480):
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    # 1. Rectangle
    cv2.rectangle(canvas, (70, 70), (320, 260), (0, 130, 220), -1) # BGR Amber
    cv2.rectangle(canvas, (70, 70), (320, 260), (30, 30, 30), 3)

    # 2. Circle
    cv2.circle(canvas, (510, 165), 90, (220, 90, 40), -1) # BGR Blue
    cv2.circle(canvas, (510, 165), 90, (30, 30, 30), 3)

    # 3. Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "CV Lab Exercise #76", (70, 350), font, 1.1, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Geometric Rendering: Rectangle, Circle & Text", (70, 390),
                font, 0.65, (100, 100, 100), 1, cv2.LINE_AA)

    # Save to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cv2.imwrite(os.path.join(script_dir, "canvas_output.jpg"), canvas)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title("Exp 76: Synthetic Canvas with Geometric Primitives & Typography", fontweight='bold')
    ax.axis('off')

    output_path = os.path.join(script_dir, "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp76] Output saved at: {output_path}")

if __name__ == "__main__":
    render_shapes_canvas()
