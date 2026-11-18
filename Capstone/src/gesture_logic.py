def is_open(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def detect_gesture(lm):

    if lm is None:
        return "IDLE"

    index  = is_open(lm, 8, 6)
    middle = is_open(lm, 12, 10)
    ring   = is_open(lm, 16, 14)
    pinky  = is_open(lm, 20, 18)

    if index and middle and ring and pinky:
        return "JUMP"

    if not index and not middle and not ring and not pinky:
        return "ROLL"

    if index and not middle and not ring and not pinky:
        if lm[8].x < lm[5].x - 0.03:
            return "LEFT"
        if lm[8].x > lm[5].x + 0.03:
            return "RIGHT"

    return "IDLE"
