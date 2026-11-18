import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Opened:", cap.isOpened())

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False, 1, 1, 0.5, 0.5)

draw = mp.solutions.drawing_utils

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        for h in res.multi_hand_landmarks:
            draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("MediaPipe Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
