import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import numpy as np
import cvzone
from pynput.keyboard import Controller, Key

cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=2)
keyList = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "del"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ":"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", "."],
]

keyboard = Controller()


class Button:
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.text = text
        self.size = size


def drawKey(img, pos, size, color, text, textPos, textSize):
    """
    Draw a single key
    """

    cv2.rectangle(img, pos, size, color, cv2.FILLED)
    cv2.putText(img, text, textPos, cv2.FONT_HERSHEY_PLAIN, textSize, (0, 0, 0), 3)


def drawAll(img, btnList):
    """
    Put all keyboard keys to the image
    """

    imgNew = np.zeros_like(img, np.uint8)

    for btn in btnList:
        x, y = btn.pos
        w, h = btn.size
        cvzone.cornerRect(
            imgNew, (btn.pos[0], btn.pos[1], btn.size[0], btn.size[1]), 0, rt=0
        )
        drawKey(
            imgNew,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            btn.text,
            (x + 40, y + 60),
            2,
        )

    out = img.copy()
    mask = imgNew.astype(bool)
    alpha = 0.2
    out[mask] = cv2.addWeighted(out, alpha, imgNew, 1 - alpha, 0)[mask]
    return out


def main():

    # Create buttons list
    btnList = []
    for row in range(len(keyList)):
        for x, key in enumerate(keyList[row]):
            btnList.append(
                Button(
                    [((x + 1) * 100 + ((row + 1) * 50)) - 75, (row * 100 + 350)], key
                )
            )

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        img = detector.findHands(img)
        lmList, bboxInfo = detector.findPosition(img)
        img = drawAll(img, btnList)

        if lmList:
            for btn in btnList:
                x, y = btn.pos
                w, h = btn.size

                # Check if our index finger is in a key area
                if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                    drawKey(
                        img,
                        (x, y),
                        (x + w, y + h),
                        (255, 200, 0),
                        btn.text,
                        (x + 25, y + 60),
                        3,
                    )

                    length, _, _ = detector.findDistance(8, 12, img, draw=False)

                    # Check if our index finger and middle finger distance is close. If yes, then write
                    if length < 40:
                        drawKey(
                            img,
                            (x, y),
                            (x + w, y + h),
                            (10, 255, 0),
                            btn.text,
                            (x + 25, y + 60),
                            3,
                        )

                        if btn.text == "del":
                            keyboard.press(Key.backspace)
                            keyboard.release(Key.backspace)
                        else:
                            keyboard.press(btn.text)
                            keyboard.release(btn.text)

                        time.sleep(0.2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
