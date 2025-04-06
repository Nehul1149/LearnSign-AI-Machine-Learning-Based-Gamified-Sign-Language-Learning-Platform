import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#
# import cv2
# from src.detector import HandDetector
# from src.recognizer import SignRecognizer
#
# detector = HandDetector()
# recognizer = SignRecognizer("models/sign_model.h5")
#
# cap = cv2.VideoCapture(0)
#
# while True:
#     success, img = cap.read()
#     if not success:
#         break
#
#     img = detector.detect_hands(img)
#
#     # Here you would extract hand landmarks and pass them to the recognizer
#     # hand_data = extract_landmark_data(img)
#     # sign = recognizer.predict(hand_data)
#
#     cv2.imshow("Sign Language Recognition", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


import cv2
from src.detector import HandDetector
from src.recognizer import SignRecognizer

print("Initializing Hand Detector...")
detector = HandDetector()
print("Hand Detector initialized.")

print("Initializing Sign Recognizer...")
recognizer = SignRecognizer("models/sign_model.h5")
print("Sign Recognizer initialized.")

print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam opened successfully.")

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame from webcam.")
        break

    img = detector.detect_hands(img)

    # Extract and predict (commented for now)
    # hand_data = extract_landmark_data(img)
    # sign = recognizer.predict(hand_data)

    cv2.imshow("Sign Language Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
