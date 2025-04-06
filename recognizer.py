# Sign language recognition module

import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import time

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Fancy Loading Screen
print("\n🚀 Initializing Sign Language Recognizer...\n")
time.sleep(2)  # Simulated loading effect

# Load the trained model
try:
    print("✅ Loading model...")
    model_path = "models/sign_model.keras"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: Model file '{model_path}' not found!")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print(f"❌ Critical Error: {e}")
    exit()

# Load the label encoder
try:
    print("🔠 Loading label encoder...")
    label_encoder_path = "models/label_encoder.npy"
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Error: Label encoder file '{label_encoder_path}' not found!")
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    print("✅ Label encoder loaded successfully!\n")
except Exception as e:
    print(f"❌ Critical Error: {e}")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open Webcam
print("🎥 Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Could not open webcam.")

print("✅ Webcam opened successfully. Starting recognition...\n")

# Caption Storage
caption = ""
last_prediction = None
prediction_time = time.time()  # Timestamp to control caption update speed

# Colors and Font
TEXT_COLOR = (0, 0, 255)  # Red for prediction text
BG_COLOR = (0, 0, 0, 150)  # Semi-transparent black background for caption
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Frame Capture Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture frame.")
        continue

    frame = cv2.flip(frame, 1)  # Mirror effect
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract 21 landmark points
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().reshape(1, -1)

            # Make prediction
            prediction = model.predict(landmarks)
            confidence = np.max(prediction) * 100  # Convert to percentage
            predicted_label = np.argmax(prediction)
            predicted_letter = label_encoder.inverse_transform([predicted_label])[0]

            # Set skeleton color based on confidence
            if confidence > 70:
                skeleton_color = (0, 255, 0)  # Green 🟢 for high confidence
            else:
                skeleton_color = (0, 0, 255)  # Red 🔴 for low confidence

            # Display Hand Skeleton with thin lines
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=skeleton_color, thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=skeleton_color, thickness=2))

            # Show Prediction at Top
            cv2.putText(frame, f"Prediction: {predicted_letter} ({confidence:.1f}%)", (50, 50),
                        FONT, 1, TEXT_COLOR, 2, cv2.LINE_AA)

            # Control caption update speed
            if predicted_letter != last_prediction:
                prediction_time = time.time()  # Reset timer
                last_prediction = predicted_letter

            # Add to Caption after 2 seconds of stability
            if time.time() - prediction_time > 2:
                if predicted_letter == "space":
                    caption += " "
                elif predicted_letter == "del":
                    caption = caption[:-1]
                elif predicted_letter not in ["nothing"]:
                    caption += predicted_letter
                prediction_time = time.time()  # Reset timer to avoid repeated fast updates

    # Draw Caption Box
    overlay = frame.copy()
    cv2.rectangle(overlay, (50, 400), (600, 450), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Display Caption
    cv2.putText(frame, caption, (60, 440), FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the Frame
    cv2.imshow("Sign Language Recognition", frame)

    # ✅ Screenshot & Save Feature
    if cv2.waitKey(1) & 0xFF == ord("c"):
        # Create filename with recognized caption
        if caption.strip():
            filename = f"captured_{caption.replace(' ', '_')}.png"
        else:
            filename = "captured_sign.png"

        # Save the frame
        cv2.imwrite(filename, frame)
        print(f"✅ Screenshot saved as '{filename}'")

    # Exit Condition
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
