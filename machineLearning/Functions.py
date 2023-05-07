import numpy as np
import math
from ModelImporter import ImportModel
import copy


# Write your ML model's name in the quotation marks here!
file_name = "MLmodel.pickle"







def pose_classification(localWithRotation, globalHeightList, velList, desired_points):
    '''A function that will take the data captured by the camera in SkeletonTracking.py, then
    classify it with the ML model.

    Parameters:

    localWithRotation - 2D list
    Contains the id, localX in meters, localY in meters, localZ in meters with origin being 
    middle of hips and rotated so the X-axis is always in line with the hips and the Y-axis 
    is in plane with the upper body.

    globalHeightList - 2D list
    Same as localList but with the Y values shifted down with respect to the lower ankle.

    desired_points - list
    Contains the indices of the anatomical points we want to record.
    '''

    info = []

    for i in desired_points:
        info.append(localWithRotation[i][1]) # Local X
        info.append(localWithRotation[i][2]) # Local Y
        info.append(localWithRotation[i][3]) # Local Z
        info.append(globalHeightList[i][2]) # Height
        info.append(velList[i][1])


    # Taking the model and using it to predict pose (to change the model, change file_name on line 10)
    try:
        clf = ImportModel(file_name)
        data = np.array([info])
        result = str(clf.predict(data)[0])
    except FileNotFoundError:
        raise Exception("Model not found. Please ensure that the file name on line 11 of Functions.py is correct.")

    return result







def globalHeightList(localList):
    '''returns all the landmarks in localList with all of the Y values 
    shifted down with Y=0 being whichever ankle is lower.
    
    Parameters:
    localList - the list with x, y, z in meters with origin being in between hips
    '''
    
    globalHeightList = copy.deepcopy(localList)
    minHeight = 0
    if localList[27][2] < minHeight:
        minHeight = localList[27][2]
    if localList[28][2] < minHeight:
        minHeight = localList[28][2]

    for i in range(len(globalHeightList)):
        globalHeightList[i][2] = globalHeightList[i][2] - minHeight
    
    return globalHeightList

def convert_local_to_localWithRotation(localList):
    V1 = np.array([localList[23][1], localList[23][2], localList[23][3]])
    V2 = np.array([localList[11][1], localList[11][2], localList[11][3]])
    V3 = np.cross(V1, V2)
    V4 = np.cross(V3, V1)

    X = V1 / np.linalg.norm(V1)
    Y = V4 / np.linalg.norm(V4)
    Z = V3 / np.linalg.norm(V3)


    

    R = np.array([[X[0], Y[0], Z[0]],
                  [X[1], Y[1], Z[1]],
                  [X[2], Y[2], Z[2]],])

    Rinv = np.linalg.inv(R)
    
    
    localWithRotationList = copy.deepcopy(localList)

    

    for i in localWithRotationList:
        pointXYZ = np.array([i[1], i[2], i[3]]).T
        rotated = Rinv.dot(pointXYZ)
        i[1] = rotated[0]
        i[2] = rotated[1]
        i[3] = rotated[2]

    return localWithRotationList

def full_column_names(desired_points):
    '''Turns the desired anatomical points into a list of fitting column names.'''
    
    columns = []
    
    # Defining column names (there will be 76 of them)
    for i in desired_points:
        columns.append(str(i)+": X") # Local with Rotation X value 
        columns.append(str(i)+": Y") # Local with Rotation Y value
        columns.append(str(i)+": Z") # Local with Rotation Z value
        columns.append(str(i)+": Height") # Height from the "floor" (Lowest landmark is "0" height)
        columns.append(str(i)+": Velocity") # Velocity of the landmark (with reference to middle of hips)
    columns.append("Results")
    return columns



def velocity(previous_localList, current_localList, dt):
    velList = []

    if len(previous_localList) != 0 and len(current_localList) != 0: # if both frames have a body detected
        for i in range(len(previous_localList)):
            x = current_localList[i][1] - previous_localList[i][1]
            y = current_localList[i][2] - previous_localList[i][2]
            z = current_localList[i][3] - previous_localList[i][3]

            velocity = math.sqrt(x*x + y*y + z*z) / dt
            velList.append([i, velocity])
    
    return velList

        
    