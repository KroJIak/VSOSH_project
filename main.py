from handWorking import handDetector, globalHandWorker
from database import dbWorker
from arduino import Arduino
import cv2

ARDUINO_PORT = '/dev/ttyUSB0'

arduino = Arduino(ARDUINO_PORT, baudrate=9600, log=False, timeout=1)
detector = handDetector(detectionCon=0.8, maxHands=2)
handWorker = globalHandWorker()
db = dbWorker('database.json')
dbData = db.get()
glAngle = 90
turn = True

def searchServo():
    global turn, glAngle
    if turn: glAngle += 1
    else: glAngle -= 1
    arduino.setAngle(glAngle)
    if glAngle >= 180: turn = False
    if glAngle <= 0: turn = True


def main():
    global glAngle
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 30)

    needGestureName = input('Введите название жеста:\n>>> ')
    while True:
        success, img = cap.read()
        width = len(img[0])
        hands = detector.findHands(img)
        if hands:
            onlyMain = handWorker.getOnlyMainHands(hands)
            _, gestureName, _, _ = handWorker.getMaxPossibleGesture(onlyMain, None, dbData)
            print(f'Найденный жест: {gestureName}')
            centerX = [hand['lmList'][9]['x'] for typeHand, hand in onlyMain.items()]
            centerX = sum(centerX) / len(onlyMain)
            kp = 0.003
            error = (width // 2) - centerX
            u = error * kp
            if abs(u) > 1:
                glAngle += u
                glAngle = max(0, min(180, glAngle))
                arduino.setAngle(glAngle)
            if gestureName == needGestureName: arduino.setGreenLight()
            else: arduino.setRedLight()
        else:
            arduino.setRedLight()
            searchServo()

if __name__ == '__main__':
    main()
arduino.setAngle(0)