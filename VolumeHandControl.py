import math

import cv2
import time
import numpy as np
import HandTrackingModule as htm
import osascript as osa

############################
wCam, hCam = 1280, 720
############################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Volume Settings:
minVol = 0
maxVol = 100

pTime = 0

detector = htm.handDetector(maxHands=1, detectionCon = 0.7)

while True:

    # result = osa.osascript('get volume settings')
    # print(result)
    # print(type(result))
    # volInfo = result[1].split(',')
    # outputVol = volInfo[0].replace('output volume:', '')
    # print(outputVol)

    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 255, 0), cv2.FILLED)

        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        # print(length)

        # Length Range: 50-300
        # Volume Ramge: 0- 100

        vol = np.interp(length, [50, 300], [minVol, maxVol])

        print(int(vol))

        conVol = "set volume output volume " + str(int(vol))

        osa.osascript(conVol)


        if length<50:
            cv2.circle(img, (cx, cy), 15, (0, 0, 55), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Img", img)
    cv2.waitKey(1)


