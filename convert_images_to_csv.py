# import os
# import cv2
# import mediapipe as mp
# import pandas as pd
#
# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
#
# # Dataset Path (Modify this to your dataset location)
# DATASET_PATH = "datasets"  # Example: "datasets/"
# CSV_FILE_PATH = "hand_landmarks.csv"  # Output CSV file
#
# # List to store landmark data
# data = []
#
# # Loop through each subfolder (A, B, C, ...)
# for label in sorted(os.listdir(DATASET_PATH)):
#     label_path = os.path.join(DATASET_PATH, label)
#
#     # Check if it's a directory
#     if os.path.isdir(label_path):
#         print(f"Processing letter: {label}")
#
#         # Loop through all images in the letter subfolder
#         for image_name in os.listdir(label_path):
#             image_path = os.path.join(label_path, image_name)
#
#             # Load image
#             image = cv2.imread(image_path)
#             if image is None:
#                 print(f"Could not load {image_path}")
#                 continue
#
#             # Convert to RGB (MediaPipe requires RGB format)
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#             # Process image with MediaPipe Hands
#             result = hands.process(image_rgb)
#
#             # If hand landmarks are detected
#             if result.multi_hand_landmarks:
#                 for hand_landmarks in result.multi_hand_landmarks:
#                     # Store (x, y, z) for all 21 landmarks
#                     landmark_list = []
#                     for lm in hand_landmarks.landmark:
#                         landmark_list.extend([lm.x, lm.y, lm.z])  # Append x, y, z values
#
#                     # Append label (A, B, C, etc.)
#                     landmark_list.append(label)
#
#                     # Save to data list
#                     data.append(landmark_list)
#
# # Convert to DataFrame
# columns = [f"x_{i}, y_{i}, z_{i}" for i in range(21)]  # Landmark feature names
# columns.append("label")  # Last column is the label (A, B, C, etc.)
#
# df = pd.DataFrame(data, columns=columns)
#
# # Save to CSV
# df.to_csv(CSV_FILE_PATH, index=False)
#
# print(f" Hand landmark dataset saved to {CSV_FILE_PATH}")


import os
import cv2
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define dataset path and CSV output file
DATASET_PATH = "datasets"  # Change this to your dataset folder path
CSV_OUTPUT = "hand_gestures.csv"

# Prepare data storage
data = []

# Iterate through each alphabet folder (A, B, C, etc.)
for letter_folder in sorted(os.listdir(DATASET_PATH)):
    letter_path = os.path.join(DATASET_PATH, letter_folder)

    if not os.path.isdir(letter_path):
        continue  # Skip files, only process folders

    # Check if it's a directory
    if os.path.isdir(letter_path):
        print(f"Processing letter: {letter_folder}")

    # Iterate through images inside each letter folder
    for image_file in os.listdir(letter_path):
        image_path = os.path.join(letter_path, image_file)

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            continue  # Skip unreadable images

        # Convert image to RGB (MediaPipe requires RGB format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands
        result = hands.process(image_rgb)

        # If hands are detected, extract landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Extract 21 landmark (x, y, z) coordinates
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Append landmarks and label to data
                data.append(landmarks + [letter_folder])

# Convert to DataFrame and save as CSV
columns = [f'landmark_{i+1}_{axis}' for i in range(21) for axis in ('x', 'y', 'z')] + ['label']
df = pd.DataFrame(data, columns=columns)
df.to_csv(CSV_OUTPUT, index=False)

print(f"CSV file saved as {CSV_OUTPUT}")
