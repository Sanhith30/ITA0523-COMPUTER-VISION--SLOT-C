"""
Experiment 51: Face Detection Application with Detection Counter
Objective: Load facial Haar cascade, locate human faces, draw bounding rectangles,
and render an on-screen display banner showing the total detected face count.
"""

import os
import cv2
import matplotlib.pyplot as plt

def face_detection_counter(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    annotated = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
    count = len(faces)

    for i, (x, y, w, h) in enumerate(faces, 1):
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 220, 255), 3)
        cv2.putText(annotated, f"Face #{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

    # Count overlay box
    cv2.rectangle(annotated, (20, 20), (380, 80), (20, 20, 20), -1)
    cv2.putText(annotated, f"Faces Detected: {count}", (35, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(annotated)
    axes[1].set_title(f"Face Detection Output (Total Detected: {count})", fontweight='bold')
    axes[1].axis('off')

    plt.suptitle("Exp 51: Haar Cascade Face Enumeration Application", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp51] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    face_detection_counter(os.path.join(script_dir, "input.jpg"))
