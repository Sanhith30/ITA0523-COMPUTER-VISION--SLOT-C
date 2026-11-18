import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self):

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print("Camera opened:", self.cap.isOpened())

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.drawer = mp.solutions.drawing_utils

    def get_frame_and_landmarks(self):

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None, None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = self.hands.process(rgb)

        lm = None
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            self.drawer.draw_landmarks(
                frame,
                hand,
                self.mp_hands.HAND_CONNECTIONS
            )
            lm = hand.landmark

        return frame, lm

    def release(self):
        self.hands.close()
        self.cap.release()
        cv2.destroyAllWindows()
