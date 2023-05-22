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
width = 200
height = 100
apple = cv2.resize(apple, (width, height))

#postions of image
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
        frame_height, frame_width, _ = image.shape
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
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame_width}, '
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z * frame_width}, '
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame_height})'
                # )
                IndexTipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame_width
                IndexTipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame_height
                IndexTipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z * frame_width

                ThumbTipX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame_width
                ThumbTipY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame_height
                ThumbTipZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z * frame_width


                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )


        margin = 100
        
        ThumbInRightEdge =  (ThumbTipX < x+width and ThumbTipX > x+width - margin) and (ThumbTipY < y+height and ThumbTipY > y) 
        ThumbInLeftEdge =  (ThumbTipX < x + margin and ThumbTipX > x) and (ThumbTipY < y+height and ThumbTipY > y) 
        ThumbInBottomEdge =  (ThumbTipX < x+width and ThumbTipX > x) and (ThumbTipY < y+height and ThumbTipY > y+height - margin) 
        ThumbInTopEdge =  (ThumbTipX < x+width and ThumbTipX > x) and (ThumbTipY < y + margin and ThumbTipY > y) 

        IndexInRightEdge =  (IndexTipX < x+width and IndexTipX > x+width - margin) and (IndexTipY < y+height and IndexTipY > y) 
        IndexInLeftEdge =  (IndexTipX < x + margin and IndexTipX > x) and (IndexTipY < y+height and IndexTipY > y) 
        IndexInBottomEdge =  (IndexTipX < x+width and IndexTipX > x) and (IndexTipY < y+height and IndexTipY > y+height - margin) 
        IndexInTopEdge =  (IndexTipX < x+width and IndexTipX > x) and (IndexTipY < y + margin and IndexTipY > y) 

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
            if x <= 0:
                x = 0
            if x >= frame_width - width:
                x = frame_width - width
            if y <= 0:
                y = 0
            if y >= frame_height - height:
                y = frame_height - height

        else:
            ImageMoveX = 0
            ImageMoveY = 0


        # Can work on how to improve frame rate? jerking hand will not move the image

        print("x", x)
        print("y", y)
        # Region of Image (ROI), where we want to insert logo *roi does NOT make a copy of the image array*
        roi = image[y:y+height, x:x+width] # 3D array (with color rgb) and 480X640
        #ROI SOMETIMES BUGGY when starting?
        print("roi",roi)
        print("Roi shape", roi.shape)
        print("apple", apple)

        previousFrameXaverage = currentFrameXaverage
        previousFrameYaverage = currentFrameYaverage

        # Set the ROI region to zeros
        # roi[:] = 0 # So the image is on top and not see through
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
