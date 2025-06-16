# import cv2
#
# cap = cv2.VideoCapture(0)  # Try changing to 1 if 0 doesn't work
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break
#     cv2.imshow("Webcam Test", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
#         break
#
# cap.release()
# cv2.destroyAllWindows()



# after train the module and run the recognizer.py file
import cv2

print("Testing webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame.")
        break

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam test complete.")
