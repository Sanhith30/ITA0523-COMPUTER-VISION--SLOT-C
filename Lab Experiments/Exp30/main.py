"""
Experiment 30: Synthetic Canvas with Geometric Primitives and Text
Objective: Generate a blank white canvas, draw one rectangle and one circle using
OpenCV drawing functions, add user-defined text, and export the file.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_primitives_canvas(width=700, height=450, text="Computer Vision Lab - 2026"):
    # Create white canvas
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    # 1. Draw a filled colored rectangle
    cv2.rectangle(canvas, (80, 80), (300, 260), (220, 100, 30), -1) # BGR blue-ish
    cv2.rectangle(canvas, (80, 80), (300, 260), (40, 40, 40), 3) # border

    # 2. Draw a filled circle with border
    cv2.circle(canvas, (480, 170), 85, (30, 180, 220), -1) # BGR amber/gold
    cv2.circle(canvas, (480, 170), 85, (30, 100, 150), 3) # border

    # 3. Add user-defined text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, text, (80, 350), font, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Canvas: {width}x{height} | Shapes: Rectangle + Circle", (80, 390),
                font, 0.6, (100, 100, 100), 1, cv2.LINE_AA)

    # Save output file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "canvas_primitives.jpg")
    cv2.imwrite(save_path, canvas)

    # Plot output
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title("Exp 30: White Canvas with Rectangle, Circle, and Text Annotations", fontweight='bold')
    ax.axis('off')

    output_path = os.path.join(script_dir, "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp30] Output saved at: {output_path}")

if __name__ == "__main__":
    draw_primitives_canvas()
