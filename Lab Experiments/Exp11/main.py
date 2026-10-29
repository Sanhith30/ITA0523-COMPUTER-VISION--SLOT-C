"""
Experiment 11: Facial Feature Detection (Eyes and Smiles within Detected Faces)
Objective: Detect human faces using Haar Cascades, then search within each face ROI
for eyes and smiles. Annotate detected features with colored bounding boxes.
"""

import os
import cv2
import matplotlib.pyplot as plt

def detect_face_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    annotated = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load Haar Cascade Classifiers from OpenCV data
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    face_count = len(faces)
    eye_count = 0
    smile_count = 0

    for (fx, fy, fw, fh) in faces:
        # Draw bounding box for Face (Blue)
        cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (0, 100, 255), 3)
        cv2.putText(annotated, "Face", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        # Region of Interest for nested search
        roi_gray = gray[fy:fy + fh, fx:fx + fw]
        roi_color = annotated[fy:fy + fh, fx:fx + fw]

        # Detect eyes within upper 60% of face
        roi_eyes_gray = roi_gray[:int(fh * 0.65), :]
        roi_eyes_color = roi_color[:int(fh * 0.65), :]
        eyes = eye_cascade.detectMultiScale(roi_eyes_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_eyes_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2) # Green for eyes
            eye_count += 1

        # Detect smile within lower half of face
        roi_smile_gray = roi_gray[int(fh * 0.45):, :]
        roi_smile_color = roi_color[int(fh * 0.45):, :]
        smiles = smile_cascade.detectMultiScale(roi_smile_gray, scaleFactor=1.3, minNeighbors=10, minSize=(20, 20))
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_smile_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2) # Red for smile
            smile_count += 1
            break # display dominant smile

    # Plot original vs annotated
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(annotated)
    title = f"Detections: {face_count} Face(s) (Blue), {eye_count} Eye(s) (Green), {smile_count} Smile(s) (Red)"
    axes[1].set_title(title, fontweight='bold')
    axes[1].axis('off')

    plt.suptitle("Exp 11: Hierarchical Face, Eye, and Smile Detection", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp11] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    detect_face_features(os.path.join(script_dir, "input.jpg"))
