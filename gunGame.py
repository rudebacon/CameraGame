import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)  # Use the default camera (usually the built-in webcam)

with mp_hands.Hands(
    max_num_hands=2,  # Detect up to 2 hands in the frame
    min_detection_confidence=0.5,  # Minimum confidence threshold for detection
    min_tracking_confidence=0.5,  # Minimum confidence threshold for tracking
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB and process it with Mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand landmarks and connections on the image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Show the image with hand landmarks on the screen
        cv2.imshow("Hand Tracking", image)
        if cv2.waitKey(5) & 0xFF == 27:  # Exit if the user presses the "Esc" key
            break

cap.release()
cv2.destroyAllWindows()
