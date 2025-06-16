# # Sign language recognition module
#
# import tensorflow as tf
# import numpy as np
#
# class SignRecognizer:
#     def __init__(self, model_path):
#         self.model = tf.keras.models.load_model(model_path)
#
#     def predict(self, hand_data):
#         hand_data = np.array(hand_data).reshape(1, -1)  # Reshape for model input
#         prediction = self.model.predict(hand_data)
#         return np.argmax(prediction)  # Return the predicted class index


# import tensorflow as tf
# import numpy as np
#
# class SignRecognizer:
#     def __init__(self, model_path):
#         print(f"Loading model from {model_path}...")  # Debug print
#         try:
#             self.model = tf.keras.models.load_model(model_path)
#             print("Model loaded successfully.")  # Debug print
#         except Exception as e:
#             print(f"Error loading model: {e}")  # Debug print
#
#     def predict(self, hand_data):
#         hand_data = np.array(hand_data).reshape(1, -1)  # Reshape for model input
#         prediction = self.model.predict(hand_data)
#         return np.argmax(prediction)  # Return the predicted class index


#
#
# import tensorflow as tf
# import numpy as np
# import os
#
# class SignRecognizer:
#     def __init__(self, model_path):
#         print(f"Loading model from {model_path}...")
#         if not os.path.exists(model_path):
#             print("⚠️ Warning: Model file not found! Sign recognition will not work.")
#             self.model = None  # Avoid crashing
#             return
#
#         try:
#             self.model = tf.keras.models.load_model(model_path)
#             print("✅ Model loaded successfully.")
#         except Exception as e:
#             print(f"❌ Error loading model: {e}")
#             self.model = None
#
#     def predict(self, hand_data):
#         if self.model is None:
#             print("⚠️ No model loaded. Cannot predict.")
#             return None
#
#         hand_data = np.array(hand_data).reshape(1, -1)
#         prediction = self.model.predict(hand_data)
#         return np.argmax(prediction)


import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle

# Load the trained model
model = tf.keras.models.load_model("models/sign_model.h5")

# Load the label encoder
with open("models/label_encoder.npy", "rb") as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract 21 landmark points (x, y, z)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Convert landmarks into NumPy array for model prediction
            landmarks = np.array(landmarks).reshape(1, -1)

            # Make prediction
            prediction = model.predict(landmarks)
            predicted_label = np.argmax(prediction)  # Get class index
            predicted_letter = label_encoder.inverse_transform([predicted_label])[0]  # Convert index to letter

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the predicted letter on the frame
            cv2.putText(frame, f"Prediction: {predicted_letter}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Sign Language Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
