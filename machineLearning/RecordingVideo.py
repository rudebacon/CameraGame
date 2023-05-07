## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

## Heavily Modified By: Sager Education.

#########################################################
##      Open CV, MediaPipe, & Numpy integration        ##
##                 Skeleton Tracking                   ##
#########################################################

'''Instructions: Run this file to record a video of safe/risky/fallen poses'''


import cv2
import time
import FP_PoseModule as pm
import datetime


# Get the current date and time
now = datetime.datetime.now()

# Generate a filename with a timestamp
filename = "DataVideos/pose_video_{}.avi".format(now.strftime("%Y-%m-%d_%H-%M-%S"))





detector = pm.poseDetector()

# Set variable for FPS calculations
pTime = 0

# Only tracking shoulders, wrists, elbows, hips, knees, ankles
desired_points = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


current_localList = []

# For webcam input:
cap = cv2.VideoCapture(0)

# For recording
VIDEO_FPS = 60 # This number doesn't really matter unless you are viewing from the file itself, DataCollection.py will open this video file and go through it frame by frame anyway (So FPS doesn't matter)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(filename, fourcc, VIDEO_FPS, (640,480))




while cap.isOpened():
    success, color_image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    

    # Comment out for non-mirror view
    color_image = cv2.flip(color_image, 1)

    # Calculate and display FPS information 
    cTime = time.time()
    dt = cTime - pTime
    fps = 1 / dt
    pTime = cTime
    

    cv2.putText(color_image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 0, 255), 3)

    # Record image in video
    out.write(color_image)
    

    color_image = detector.findPose(color_image, draw=True) # Draws pipes and processes image

    lmList = detector.findPosition(color_image, draw=True) # Draws points and returns pixel coordinates of landmarks
    

    # previous_localList = current_localList # For velList
    # localList = detector.findLocalWorldLandmarks()
    # current_localList = localList # For velList


    # velList = Functions.velocity(previous_localList, current_localList, dt)
    


    # Checking to see if the desired points exist on camera
    count = 0
    for i in lmList:
        if i[0] in desired_points and 0<=i[1]<=640 and 0<=i[2]<=480: #If the point is in desired points and is in frame
            count += 1

    if count == len(desired_points):
        

        result = "All Needed Landmarks In View"
        
    else:
        result = "Not In View"



    

    # Printing the result
    cv2.putText(color_image, result, (70, 400), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)




    # Show image
    cv2.namedWindow('Webcam', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Webcam', color_image)


    if cv2.waitKey(1) == 27:
        # Stop streaming
        cv2.destroyAllWindows()
        break
# Stop occupying camera
cap.release()
out.release()

