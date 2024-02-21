from math import sqrt, acos, pi
import mediapipe as mp
import multiprocessing
import numpy as np
import cv2


PARENT_POINTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
FLIP_HAND_DICT = {
    'Right': 'Left',
    'Left': 'Right'
}

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)


    def getResults(self, imgRGB):
        self.results = self.hands.process(imgRGB)

    def findHands(self, img, flipType=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        processFinding = multiprocessing.Process(self.getResults(imgRGB))
        while processFinding.is_alive(): pass
        self.height, self.width = img.shape[:2]
        allHands = []
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                lmList = []
                for lm in handLms.landmark:
                    px, py, pz = lm.x * self.width, lm.y * self.height, lm.z * self.width
                    lmList.append(dict(x=px, y=py, z=pz))

                typeHand = handType.classification[0].label
                scoreHand = handType.classification[0].score
                if flipType: typeHand = FLIP_HAND_DICT[typeHand]
                allHands.append({
                    'lmList': lmList,
                    'type': typeHand,
                    'score': scoreHand
                })
        return allHands

class globalHandWorker():
    def getResultHands(self, realHands):
        if not realHands: return []
        resultHands = {hand['type']: {'lmList': hand['lmList'],
                                   'score': hand['score']} for hand in realHands}
        return resultHands

    def getOnlyMainHands(self, hands):
        if hands is None: return None
        newHands = {}
        for hand in hands:
            newHands[hand['type']] = {
                'lmList': hand['lmList'],
                'score': hand['score']
            }
        return newHands

    def onlyMainHands2LmList(self, hands):
        lmList = [hands[typeHand]['lmList'] for typeHand in hands]
        return lmList

    def getAngleBetweenLines(self, line1, line2):
        startPosLine1, endPosLine1 = line1
        startPosLine2, endPosLine2 = line2
        vector1 = {axis: (endPosLine1[axis] - startPosLine1[axis]) for axis in ['x', 'y', 'z']}
        vector2 = {axis: (endPosLine2[axis] - startPosLine2[axis]) for axis in ['x', 'y', 'z']}
        scalarProduct = np.dot(list(vector1.values()), list(vector2.values()))
        lengthVector1 = sqrt(vector1['x']**2 + vector1['y']**2 + vector1['z']**2)
        lengthVector2 = sqrt(vector2['x']**2 + vector2['y']**2 + vector2['z']**2)
        lengthsProduct = lengthVector1 * lengthVector2
        if lengthsProduct == 0: return pi
        angle = acos(scalarProduct / lengthsProduct)
        return angle

    def getDistanceBetweenPoints2Dimg(self, point1, point2):
        vector = {axis: (point1[axis] - point2[axis]) for axis in ['x', 'y']}
        lengthVector = sqrt(vector['x'] ** 2 + vector['y'] ** 2)
        return lengthVector

    def getPercentLinesHandSimilarity(self, lmListFullHand, lmListRealHand, numPoint):
        angle = self.getAngleBetweenLines((lmListRealHand[PARENT_POINTS[numPoint]], lmListRealHand[numPoint]),
                                                (lmListFullHand[PARENT_POINTS[numPoint]], lmListFullHand[numPoint]))
        anglePercent = (pi - angle) / pi
        return anglePercent

    def getLineHandsPercent(self, resultHands, fullGesture, resultFace=None):
        fullHands = fullGesture['hands']
        lineHandsPercent = {'Right': [1] + [0] * 20, 'Left': [1] + [0] * 20}
        for typeHand in fullHands:
            if typeHand not in resultHands: continue
            lmListRealHand = resultHands[typeHand]['lmList']
            lmListFullHand = fullHands[typeHand]['lmList']
            for point in range(1, 21):
                lineHandsPercent[typeHand][point] = self.getPercentLinesHandSimilarity(lmListFullHand,
                                                                                       lmListRealHand, point)
            if fullGesture['useFace']:
                if resultFace:
                    averageDistancePoints = []
                    for pointHand, pointFace, needDistance in fullGesture['linkedPointsWithFace']:
                        distancePoints = self.getDistanceBetweenPoints2Dimg(
                            resultHands[typeHand]['lmList'][pointHand],
                            resultFace['lmList'][pointFace])
                        distanceRatio = needDistance / distancePoints if distancePoints > 0 else 1
                        averageDistancePoints.append(min(1, distanceRatio))
                    distancePercent = sum(averageDistancePoints) / len(averageDistancePoints)
                else:
                    distancePercent = 0
                lineHandsPercent[typeHand] = np.dot(lineHandsPercent[typeHand], distancePercent)
        return lineHandsPercent

    def getResultPercent(self, hands, fullGesture, face, indexCount):
        lineHandsPercent = self.getLineHandsPercent(hands, fullGesture, face)
        if indexCount == 0:
            resultPercent = [sum(lineHandsPercent[typeHand][1:]) / (len(lineHandsPercent[typeHand]) - 1)
                             for typeHand in lineHandsPercent if typeHand in fullGesture['hands']]
            resultPercent = sum(resultPercent) / 2
        elif indexCount == 1:
            typeHand = list(hands.keys())[0]
            resultPercent = sum(lineHandsPercent[typeHand][1:]) / (len(lineHandsPercent[typeHand]) - 1)
        return resultPercent, lineHandsPercent

    def getMaxPossibleGesture(self, hands, face, dbData, oldGestureData=None):
        if oldGestureData:
            fullGesture = dbData[oldGestureData['type']]['gestures'][oldGestureData['name']]
            gestureInfo = dbData[oldGestureData['type']]['info']
            for ind, category in enumerate(gestureInfo):
                if oldGestureData['name'] in gestureInfo[category]:
                    indexCount = ind
                    break
            resultPercent, lineHandsPercent = self.getResultPercent(hands, fullGesture, face, indexCount)
            if resultPercent >= oldGestureData['percent']:
                return oldGestureData['type'], oldGestureData['name'], resultPercent, lineHandsPercent
        maxPercentList = None
        maxGestureName = None
        maxGestureType = None
        maxPercent = 0
        for gestureType in dbData:
            if not len(dbData[gestureType]): continue
            gestureInfo = dbData[gestureType]['info']
            for ind, category in enumerate(gestureInfo):
                countHandList = gestureInfo[category]
                for gestureName in countHandList:
                    fullGesture = dbData[gestureType]['gestures'][gestureName]
                    resultPercent, lineHandsPercent = self.getResultPercent(hands, fullGesture, face, ind)
                    if resultPercent > maxPercent:
                        maxPercent = resultPercent
                        maxGestureType = gestureType
                        maxGestureName = gestureName
                        maxPercentList = lineHandsPercent.copy()
        return maxGestureType, maxGestureName, maxPercent, maxPercentList
