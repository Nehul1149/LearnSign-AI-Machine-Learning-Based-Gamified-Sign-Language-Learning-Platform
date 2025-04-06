import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # To draw landmarks

# Load an image (replace with your file path)
image_path = "datasets/A/A1.jpg" # "path_to_your_image.jpg"
image = cv2.imread(image_path)

# Convert to RGB (MediaPipe requires RGB format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image with MediaPipe Hands
result = hands.process(image_rgb)

# Check if a hand is detected
if result.multi_hand_landmarks:
    for hand_landmarks in result.multi_hand_landmarks:
        print("Hand landmarks detected:")
        for i, lm in enumerate(hand_landmarks.landmark):
            print(f"Landmark {i}: (x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:.4f})")

        # Draw landmarks on the image
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# Show the image with landmarks
cv2.imshow("Hand Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
