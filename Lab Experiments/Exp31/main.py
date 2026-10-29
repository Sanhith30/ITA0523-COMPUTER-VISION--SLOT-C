"""
Experiment 31: Face Detection with Total Count (Haar Cascade)
Objective: Detect human faces in supplied image, draw bounding boxes,
count total detections, and display annotated image with count clearly visible.
"""

import os
import cv2
import matplotlib.pyplot as plt

def detect_faces_with_count(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    annotated = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
    total_faces = len(faces)

    for i, (x, y, w, h) in enumerate(faces, 1):
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(annotated, f"Face #{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Banner showing total count
    banner_text = f"Total Faces Detected: {total_faces}"
    cv2.putText(annotated, banner_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 3)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Input Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(annotated)
    axes[1].set_title(f"Face Localization Result ({total_faces} Face(s) Found)", fontweight='bold')
    axes[1].axis('off')

    plt.suptitle("Exp 31: Haar Cascade Face Detection and Enumeration", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp31] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    detect_faces_with_count(os.path.join(script_dir, "input.jpg"))
