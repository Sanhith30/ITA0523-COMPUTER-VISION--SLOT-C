import cv2
import numpy as np
from src.hand_tracking import HandTracker
from src.gesture_logic import detect_gesture
from src.game_controller import GameController

MIN_CONTOUR_AREA = 4000

def hand_present(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 30, 60])
    upper = np.array([20, 150, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False

    max_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(max_contour) > MIN_CONTOUR_AREA:
        return True

    return False


def main():

    tracker = HandTracker()
    controller = GameController()

    while True:

        frame, lm = tracker.get_frame_and_landmarks()

        if frame is None:
            continue

        real_hand = hand_present(frame)

        if lm is not None and real_hand:

            gesture = detect_gesture(lm)

            controller.perform_action(gesture)

            status = f"Gesture: {gesture}"
            color = (0,255,0)

        else:
            gesture = "NO HAND"
            status = "NO HAND DETECTED"
            color = (0,0,255)

        cv2.putText(frame, status, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

        cv2.imshow("Hand Gesture Play - Hand Only", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    tracker.release()

if __name__ == "__main__":
    main()
