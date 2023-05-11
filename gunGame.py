import cv2
import mediapipe as mp
import keyboard
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)  # Use the default camera (usually the built-in webcam)
cv2.namedWindow('Hand Tracking', cv2.WINDOW_AUTOSIZE) # Create window





# Load apple image
apple = cv2.imread('images/apple.jpg')
size = 100
apple = cv2.resize(apple, (size, size))

# Create a mask of logo
img2gray = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

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
        image_height, image_width, _ = image.shape

        # Flip the image horizontally
        image = cv2.flip(image, 1)

        # PROCESSING
        # Convert the BGR image to RGB and process it with Mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        # Draw the hand landmarks and connections on the image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                print('hand_landmarks:', hand_landmarks)
                print(
                    f'Index finger tip coordinates: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                )

                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )



        



        # Region of Image (ROI), where we want to insert logo
        roi = image[-size-10:-10, -size-10:-10]
    
        # Set an index of where the mask is
        roi[np.where(mask)] = 0
        roi += apple

        # If window closed, exit (This has to be before showing image, I'm not sure why)
        if cv2.getWindowProperty('Hand Tracking', cv2.WND_PROP_VISIBLE) < 1:
            break
        # Show image
        cv2.imshow("Hand Tracking", image)


        cv2.waitKey(1) #needed to render image?

        if keyboard.is_pressed("esc"): # break if esc key pressed during playback
            break   

cap.release()
cv2.destroyAllWindows()
