# # Hand detection module
#
# import cv2
# import mediapipe as mp
#
# class HandDetector:
#     def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
#         self.mpHands = mp.solutions.hands
#         self.hands = self.mpHands.Hands(mode, maxHands, detectionCon, trackCon)
#         self.mpDraw = mp.solutions.drawing_utils
#
#     def detect_hands(self, img):
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = self.hands.process(imgRGB)
#         if results.multi_hand_landmarks:
#             for handLms in results.multi_hand_landmarks:
#                 self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
#         return img


import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        print("Initializing MediaPipe Hands...")  # Debug print
        self.mpHands = mp.solutions.hands
        # self.hands = self.mpHands.Hands(mode, maxHands, detectionCon, trackCon)
        self.hands = self.mpHands.Hands()
        print("MediaPipe Hands Initialized.")  # Debug print
        self.mpDraw = mp.solutions.drawing_utils

    def detect_hands(self, img):
        print("Processing frame...")  # Debug print
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
