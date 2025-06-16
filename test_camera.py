# import cv2
#
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
# else:
#     print("Webcam is working.")
# cap.release()


# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#
# # import mediapipe as mp
# # print(mp.solutions.hands.Hands())
#
# print(os.path.exists("models/sign_model.h5"))

# convert images to CSV
# import os
# print(os.path.exists("hand_landmarks.csv"))  # Should print True if the file exists
#
# file_path = os.path.abspath("hand_landmarks.csv")
# print(f"Saving CSV file to: {file_path}")
