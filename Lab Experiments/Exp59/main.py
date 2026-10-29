"""
Experiment 59: Facial Feature Detection (Face, Eyes, Smile)
Objective: Locate human faces, search within face regions for eyes and smile,
and draw annotated bounding boxes around recognized features.
"""

import os
import cv2
import matplotlib.pyplot as plt

def facial_feature_localization(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    annotated = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    faces = face_cas.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))

    for (x, y, w, h) in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 120, 0), 3) # Face: Orange
        cv2.putText(annotated, "Face", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = annotated[y:y+h, x:x+w]

        # Eyes
        eyes = eye_cas.detectMultiScale(roi_gray[:int(h*0.65), :], 1.1, 3, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Smile
        smiles = smile_cas.detectMultiScale(roi_gray[int(h*0.45):, :], 1.3, 10, minSize=(20, 20))
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            break

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(annotated)
    axes[1].set_title("Localized Face & Features: Face (Orange), Eyes (Green), Smile (Red)", fontweight='bold')
    axes[1].axis('off')

    plt.suptitle("Exp 59: Haar Cascade Facial Feature Localization", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(image_path), "output.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Exp59] Output saved at: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    facial_feature_localization(os.path.join(script_dir, "input.jpg"))
