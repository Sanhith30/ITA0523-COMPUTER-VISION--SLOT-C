import pyautogui

class GameController:
    def __init__(self):
        pyautogui.FAILSAFE = True

    def perform_action(self, g):

        if g == "JUMP":
            pyautogui.press("up")
            print("ACTION: JUMP")

        elif g == "ROLL":
            pyautogui.press("down")
            print("ACTION: ROLL")

        elif g == "LEFT":
            pyautogui.press("left")
            print("ACTION: LEFT")

        elif g == "RIGHT":
            pyautogui.press("right")
            print("ACTION: RIGHT")
