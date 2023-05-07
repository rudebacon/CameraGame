## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

## Heavily Modified By: Sager Education.

#########################################################
##      Open CV, MediaPipe, & Numpy integration        ##
##                 Skeleton Tracking                   ##
#########################################################

'''
Instructions: Run this file to playback a video you recorded using "RecordingVideo.py" and assign "Safe" / "Risky" / "Fallen" results using the keyboard.
   Keys:
    s key: slow playback of video
    f key: normal playback of video
    p key: pause / play video (Can't assign values while paused)
    left arrow key: set to rewind
    right arrow key: set to normal play
    1 key: hold down to assign "Risky" 
    2 key: hold down to assign "Fallen" (If both 1 and 2 are pressed, 2 has priority)
* Remember to change the video file string that's read! *
'''

# FILENAME TO READ
VIDEO_FILE_NAME = "DataVideos\pose_video_2023-02-21_16-27-34.avi"

import pandas as pd
import cv2
import time
import FP_PoseModule as pm
import Functions
import keyboard



# Reading Video File:
cap = cv2.VideoCapture(VIDEO_FILE_NAME)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

detector = pm.poseDetector()

# Set variable for FPS calculations
pTime = 0

# Only tracking shoulders, wrists, elbows, hips, knees, ankles
desired_points = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


# Creating column names for the dataframe(s)
colnames = Functions.full_column_names(desired_points)

# Create a dataframe to store the information for later:
df = pd.DataFrame(columns = colnames)

# Initializing localList for velocity calculations
current_localList = []



pause = False
# When "p" key pressed, pause / play video playback
def p_released(event):
    global pause
    pause = not pause

keyboard.on_release_key("p", p_released) 




delay = 0
# When "s" key pressed, slow video playback
def s_released(event):
    global delay
    delay = 0.1

keyboard.on_release_key("s", s_released)

# When "f" key pressed, normal video playback
def f_released(event):
    global delay
    delay = 0

keyboard.on_release_key("f", f_released)



rewind = False
# When "left arrow" key pressed, backwards video playback
def left_arrow_released(event):
    global rewind
    rewind = True
keyboard.on_release_key("left arrow", left_arrow_released)
# When "right arrow" key pressed, normal video playback
def right_arrow_released(event):
    global rewind
    rewind = False
keyboard.on_release_key("right arrow", right_arrow_released)

cv2.namedWindow('Webcam', cv2.WINDOW_AUTOSIZE) # Create window

while True:

    success, color_image = cap.read() # Automatically reads the next frame
    if not success:
        print("End of program")
        # pause = True
        break
        
    
    # Get the frame positions (current_frame starts at 0 at beginning of video)
    next_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    # print(next_frame)
    current_frame = next_frame - 1
    # print(current_frame)
    previous_frame = current_frame - 1
    # print(previous_frame)

    

    if rewind and previous_frame >= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, previous_frame)
    if rewind and previous_frame < 0: # At beginning of video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not rewind and next_frame == total_frames - 1: # At end of video
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 2) 
    if pause:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        

    time.sleep(delay) # Delay for when "s" key pressed (slow)

    


    cv2.waitKey(1) # I have no clue why, but I need two cv2.waitKey(1) somewhere in the video loop to continue playing, or else video frames stop when keys are pressed.
    cv2.waitKey(1)

    
    
    

    # # Comment out for non-mirror view
    # color_image = cv2.flip(color_image, 1)

    # Calculate and display FPS information 
    cTime = time.time()
    dt = cTime - pTime
    fps = 1 / dt
    pTime = cTime

    cv2.putText(color_image, str(int(fps)), (370, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 0, 255), 3)

    color_image = detector.findPose(color_image, draw=True) # Draws pipes and processes image

    lmList = detector.findPosition(color_image, draw=True) # Draws points and returns pixel coordinates of landmarks
    

    previous_localList = current_localList
    localList = detector.findLocalWorldLandmarks()
    current_localList = localList


    velList = Functions.velocity(previous_localList, current_localList, dt)


    # Checking to see if the desired points exist on camera
    count = 0
    for i in lmList:
        if i[0] in desired_points and 0<=i[1]<=640 and 0<=i[2]<=480: #If the point is in desired points and is in frame
            count += 1

    if count == len(desired_points) and len(velList) != 0:
        
        
        localWithRotation = Functions.convert_local_to_localWithRotation(localList)
        cv2.circle(color_image, (lmList[15][1], lmList[15][2]), 3, (0, 0, 255), -1)
        cv2.putText(color_image, str(localWithRotation[15][2]), (lmList[15][1], lmList[15][2]), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
    

        globalHeightList = Functions.globalHeightList(localList)

        rowOfFeatureValues = []
        
        # For all the desired landmarks we want to track
        for i in desired_points:

            # For adding the feature values to the PoseDataForTraining
            rowOfFeatureValues.append(localWithRotation[i][1])
            rowOfFeatureValues.append(localWithRotation[i][2])
            rowOfFeatureValues.append(localWithRotation[i][3])
            rowOfFeatureValues.append(globalHeightList[i][2])
            rowOfFeatureValues.append(velList[i][1])
        
        
        if keyboard.is_pressed("1") or keyboard.is_pressed("2"):
            if keyboard.is_pressed("2"):
                rowOfFeatureValues.append("Fallen") # add to data
                # Printing the result
                cv2.putText(color_image, "Fallen", (70, 470), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            else:
                rowOfFeatureValues.append("Risky") # add to data
                # Printing the result
                cv2.putText(color_image, "Risky", (70, 470), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        else:
            rowOfFeatureValues.append("Safe") # add to data
            # Printing the result
            cv2.putText(color_image, "Safe", (70, 470), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


        

        #Update Dataframe (AddingPoseData)
        
        df.loc[current_frame] = rowOfFeatureValues
        

        result = "Collecting Data"
        
    else:
        # All desired points not in frame
        rowOfFeatureValues = []

        for i in desired_points:
            try: 
                localWithRotation = Functions.convert_local_to_localWithRotation(localList)
                rowOfFeatureValues.append(localWithRotation[i][1])
                rowOfFeatureValues.append(localWithRotation[i][2])
                rowOfFeatureValues.append(localWithRotation[i][3])

                globalHeightList = Functions.globalHeightList(localList)
                rowOfFeatureValues.append(globalHeightList[i][2])

            except IndexError: # localList empty so couldn't convert to localWithRotation
                rowOfFeatureValues.append(-1)
                rowOfFeatureValues.append(-1)
                rowOfFeatureValues.append(-1)
                rowOfFeatureValues.append(-1)
            
            try:
                velList = Functions.velocity(previous_localList, current_localList, dt)
                rowOfFeatureValues.append(velList[i][1])
            except IndexError:
                rowOfFeatureValues.append(-1)

                
        rowOfFeatureValues.append("Cannot Determine") # All desired points aren't in frame, so just assign "Cannot Determine"
        cv2.putText(color_image, "Cannot Determine", (70, 470), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        

        #Update Dataframe (AddingPoseData)
        df.loc[current_frame] = rowOfFeatureValues
        

        result = "Collecting Empty Data"




    

    # Printing the result
    cv2.putText(color_image, result, (70, 400), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    # If video is at beginning, end, or paused, display it on frame
    if pause:
        cv2.putText(color_image, "Video Paused", (220, 240), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    elif not rewind and next_frame == total_frames - 1:
        cv2.putText(color_image, "End of Video", (220, 240), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    elif rewind and previous_frame < 0:
        cv2.putText(color_image, "Beginning of Video", (220, 240), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    

    # If window closed, exit (This has to be before showing image, I'm not sure why)
    if cv2.getWindowProperty('Webcam', cv2.WND_PROP_VISIBLE) < 1:
        break

    # Show image
    cv2.imshow('Webcam', color_image)

    if keyboard.is_pressed("esc"): # break if esc key pressed during playback
        # Stop streaming
        cv2.destroyAllWindows()
        break   
        

    


# Stop occupying video
cap.release()



# Export an Excel file with the data we collected in the df dataframe

try:
    df.to_excel("AddingPoseData.xlsx", sheet_name='sheet1', index=False)
except PermissionError:
    print("Make sure 'AddingPoseData.xlsx' window is closed so we can overwrite the data")
