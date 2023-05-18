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

x = 300
y = 300

#Start thumb tip coordinate (in pixels) in top left of image
ThumbTipX = 0
ThumbTipY = 0
ThumbTipZ = 0

#Start index tip coordinate (in pixels) in top left of image
IndexTipX = 0
IndexTipY = 0
IndexTipZ = 0

ImageMoveX = 0
ImageMoveY = 0


frame = 0
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
        frame += 1

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

                
                # print(
                #     f'Index finger tip coordinates: (',
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width}, '
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z * image_width}, '
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height})'
                # )
                IndexTipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
                IndexTipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
                IndexTipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z * image_width

                ThumbTipX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width
                ThumbTipY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height
                ThumbTipZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z * image_width


                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )


        margin = 30
        
        ThumbInRightEdge =  (ThumbTipX < x+size + margin and ThumbTipX > x+size - margin) and (ThumbTipY < y+size + margin and ThumbTipY > y - margin) 
        ThumbInLeftEdge =  (ThumbTipX < x + margin and ThumbTipX > x - margin) and (ThumbTipY < y+size + margin and ThumbTipY > y - margin) 
        ThumbInBottomEdge =  (ThumbTipX < x+size + margin and ThumbTipX > x - margin) and (ThumbTipY < y+size + margin and ThumbTipY > y+size - margin) 
        ThumbInTopEdge =  (ThumbTipX < x+size + margin and ThumbTipX > x - margin) and (ThumbTipY < y + margin and ThumbTipY > y - margin) 

        IndexInRightEdge =  (IndexTipX < x+size + margin and IndexTipX > x+size - margin) and (IndexTipY < y+size + margin and IndexTipY > y - margin) 
        IndexInLeftEdge =  (IndexTipX < x + margin and IndexTipX > x - margin) and (IndexTipY < y+size + margin and IndexTipY > y - margin) 
        IndexInBottomEdge =  (IndexTipX < x+size + margin and IndexTipX > x - margin) and (IndexTipY < y+size + margin and IndexTipY > y+size - margin) 
        IndexInTopEdge =  (IndexTipX < x+size + margin and IndexTipX > x - margin) and (IndexTipY < y + margin and IndexTipY > y - margin) 

        # print("TL:", IndexInLeftEdge)
        # print("TR:", IndexInRightEdge)
        # print("TB:", IndexInBottomEdge)
        # print("TT:", IndexInTopEdge)

        currentFrameXaverage = int((IndexTipX + ThumbTipX) / 2 )
        currentFrameYaverage = int((IndexTipY + ThumbTipY) / 2 )

        if frame == 1:
            previousFrameXaverage = currentFrameXaverage
            previousFrameYaverage = currentFrameYaverage


        if (ThumbInRightEdge or ThumbInLeftEdge or ThumbInBottomEdge or ThumbInTopEdge) and (IndexInBottomEdge or IndexInLeftEdge or IndexInTopEdge or IndexInRightEdge):
            ImageMoveX = currentFrameXaverage - previousFrameXaverage
            ImageMoveY = currentFrameYaverage - previousFrameYaverage
            x += ImageMoveX
            y += ImageMoveY
        else:
            ImageMoveX = 0
            ImageMoveY = 0




        print("x", x)
        print("y", y)
        # Region of Image (ROI), where we want to insert logo *roi does NOT make a copy of the image array*
        roi = image[y:y+size, x:x+size] # 3D array (with color rgb) and 480X640
        #ROI SOMETIMES BUGGY
        print("roi",roi)
        print("apple", apple)

        previousFrameXaverage = currentFrameXaverage
        previousFrameYaverage = currentFrameYaverage

        # Set the ROI region to zeros
        roi[:] = 0 # So the image is on top and not see through
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
