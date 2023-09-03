
import cv2

# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from typing import Tuple, Union
import math
import cv2
import keyboard
import time
import random
from PIL import Image



MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red



FACEMESH_LIPS = frozenset([
                           (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), 
                           (17, 314), (314, 405), (405, 321), (321, 375), (375, 291), 

                           (61, 185), (185, 40), (40, 39), (39, 37),(37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                          
                          # (291, 409),
                          # (409, 270),
                          # (270, 269),
                          # (269, 267),  
                          # (267, 0),
                          # (0, 37), 
                          # (37, 39),
                          # (39, 40), 
                          # (40, 185),
                          # (185, 61),  
                            
                          #  (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                          #  (14, 317), (317, 402), (402, 318), (318, 324), (324, 308), 

                          #  (78, 191), (191, 80), (80, 81), (81, 82),(82, 13), 
                          #  (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
                           ])


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    
    face_landmarks = face_landmarks_list[idx]
    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    # solutions.drawing_utils.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp.solutions.drawing_styles
    #     .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=FACEMESH_LIPS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    # solutions.drawing_utils.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_IRISES,
    #       landmark_drawing_spec=None,
    #       connection_drawing_spec=mp.solutions.drawing_styles
    #       .get_default_face_mesh_iris_connections_style())

  return annotated_image





def get_lip_landmark_coordinates(rgb_image, detection_result):
    # Initialize an empty list to store lip landmark coordinates
    lip_landmark_coordinates = []

    # Get the width and height of the image
    image_height, image_width, _ = rgb_image.shape

    # Loop through the detected faces to extract lip landmark coordinates
    for idx in range(len(detection_result.face_landmarks)): # Don't really need cause only one face can be detected anyway
        face_landmarks = detection_result.face_landmarks[idx]
        
        for connection in FACEMESH_LIPS:
            point1_index, point2_index = connection
            x1 = face_landmarks[point1_index].x
            y1 = face_landmarks[point1_index].y
            x2 = face_landmarks[point2_index].x
            y2 = face_landmarks[point2_index].y
            lip_landmark_coordinates.append((point1_index, int(x1 * image_width), int(y1 * image_height)))
            lip_landmark_coordinates.append((point2_index, int(x2 * image_width), int(y2 * image_height)))
        
        lip_landmark_coordinates = set(lip_landmark_coordinates)
        

    return lip_landmark_coordinates

def calculate_polygon_area(points_set):
    # Convert the set of points to a list
    points_list = list(points_set)
    
    n = len(points_list)
    area = 0.0

    for i in range(n):
        x1, y1 = points_list[i]
        x2, y2 = points_list[(i + 1) % n]
        area += (x1 * y2) - (x2 * y1)

    area = abs(area) / 2.0
    return area




# STEP 2: Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)  # Use the default camera (usually the built-in webcam)
cv2.namedWindow('Catch Apples', cv2.WINDOW_AUTOSIZE) # Create window


class Apple:
  # Load apple image
  appleWidth = 50
  appleHeight = 50
  appleImg = cv2.resize(cv2.imread('images/apple.png', cv2.IMREAD_UNCHANGED), (appleWidth, appleHeight))

  existingApples = []

  def __init__(self, xPos, yPos, fallSpeed):
    self.xPos = xPos
    self.yPos = yPos
    self.fallSpeed = fallSpeed
    self.existingApples.append(self)
    # print("Apple created: ", self)
    # print(Apple.existingApples)

  
  def fall(self, imageHeight):
    # print("The object is:", self)
    if self.yPos >= imageHeight:
      self.to_be_removed = True  # Mark for removal    
    else:
      # print("Fell a bit")
      self.yPos = self.yPos + self.fallSpeed    
  
  def roi(self, annotated_image):
    if self.yPos > annotated_image.shape[0] - Apple.appleHeight: # When image is cut-off at the bottom
      cropped_image = Apple.appleImg[0:annotated_image.shape[0] - self.yPos, :]
      # Region of Image (ROI), where we want to insert logo/image *roi does NOT make a copy of the image array*
      roi = annotated_image[self.yPos: , self.xPos:self.xPos+Apple.appleWidth] # 3D array (with color rgb) and 480X640
      # Separate the alpha channel
      alpha_channel = cropped_image[:, :, 3]
      # Create a mask for the transparent regions
      mask = alpha_channel == 0
      # Invert the mask to represent non-transparent regions
      inverse_mask = np.invert(mask)
      # Overlay the image onto the ROI using the mask
      roi[inverse_mask] = cropped_image[:, :, :3][inverse_mask]  # Apply only RGB channels from the image
      # Update the background image with the modified ROI
      annotated_image[self.yPos: , self.xPos:self.xPos+Apple.appleWidth] = roi
      
    elif self.yPos < 0: # When image is cut-off at the top
      cropped_image = Apple.appleImg[self.yPos * -1:, :]
      # Region of Image (ROI), where we want to insert logo/image *roi does NOT make a copy of the image array*
      roi = annotated_image[0:Apple.appleHeight + self.yPos, self.xPos:self.xPos+Apple.appleWidth] # 3D array (with color rgb) and 480X640
      # Separate the alpha channel
      alpha_channel = cropped_image[:, :, 3]
      # Create a mask for the transparent regions
      mask = alpha_channel == 0
      # Invert the mask to represent non-transparent regions
      inverse_mask = np.invert(mask)
      # Overlay the image onto the ROI using the mask
      roi[inverse_mask] = cropped_image[:, :, :3][inverse_mask]  # Apply only RGB channels from the image
      # Update the background image with the modified ROI
      annotated_image[0:Apple.appleHeight + self.yPos, self.xPos:self.xPos+Apple.appleWidth] = roi
    else: # Image is in middle of screen
      # Region of Image (ROI), where we want to insert logo/image *roi does NOT make a copy of the image array*
      roi = annotated_image[self.yPos:self.yPos+Apple.appleHeight, self.xPos:self.xPos+Apple.appleWidth] # 3D array (with color rgb) and 480X640
      # Separate the alpha channel
      alpha_channel = Apple.appleImg[:, :, 3]
      # Create a mask for the transparent regions
      mask = alpha_channel == 0
      # Invert the mask to represent non-transparent regions
      inverse_mask = np.invert(mask)
      # Overlay the image onto the ROI using the mask
      roi[inverse_mask] = Apple.appleImg[:, :, :3][inverse_mask]  # Apply only RGB channels from the image
      # Update the background image with the modified ROI
      annotated_image[self.yPos:self.yPos+Apple.appleHeight, self.xPos:self.xPos+Apple.appleWidth] = roi

       
       
     
     
last_apple_creation_time = time.time() # For creating apple in a time interval

while cap.isOpened():
    success, image = cap.read()
    
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    image = cv2.flip(image, 1) # mirror image horizontally
    # STEP 3: Load the input image.
    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)


    # STEP 4: Detect faces in the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    image_copy = np.copy(image.numpy_view())
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    
    
    # Example usage:
    # Call the function to get lip landmark coordinates
    lip_coordinates = get_lip_landmark_coordinates(annotated_image, detection_result) # lip_coordinates is a set: [(landmark_number, pixel_x, pixel_y), (landmark_number, pixel_x, pixel_y), etc.]
    # # Calculate the area enclosed by the lips
    # lip_area = calculate_polygon_area(lip_coordinates)
    # # Calculate the centroid of the lip landmarks
    # centroid_x = sum(x for x, _ in lip_coordinates) / len(lip_coordinates)
    # centroid_y = sum(y for _, y in lip_coordinates) / len(lip_coordinates)
    # THIS IS THE SEQUENCE OF THE INDICES  
    sequence = (61, 146, 91, 181, 84, 17 
                          , 314, 405 , 321, 375, 
                          
                          291, 409,
                          
                          270, 269,
                          
                          267, 0,
                          
                          37, 39,
                          
                          40, 185)
    testSet = []
    for i in sequence: #Making list with sequence from the set
       for j in lip_coordinates:
          if j[0] == i:
             testSet.append((j[1], j[2]))

    # print(testSet)
    # Sort the lip landmarks based on x-coordinate (you may need a more sophisticated ordering logic)
    lip_landmarks_np = np.array(testSet, dtype=np.int32)
    # Sort the lip landmarks based on x-coordinate (ascending order)
    # sorted_indices = np.argsort(lip_landmarks_np[:, 0])
    # lip_landmarks_np = lip_landmarks_np[sorted_indices]
    cv2.fillPoly(annotated_image, [lip_landmarks_np], (255, 0, 0))

# ---------------------------------------------------------------
    current_time = time.time()
    if current_time - last_apple_creation_time >= 0.5:
      new_apple = Apple(random.randint(0, image.width - Apple.appleWidth), -Apple.appleHeight, 20)
      last_apple_creation_time = current_time
# ---------------------------------------------------------------
    for i in Apple.existingApples:
      i.fall(image.height)
      # print("Apple:", i, "yPos: ", i.yPos)
    # Create a list to store apples that need to be removed
    apples_to_remove = []
    # Check for apples marked for removal and remove them
    for i in Apple.existingApples:
        if hasattr(i, 'to_be_removed'):
            apples_to_remove.append(i)
            # print("Apples to be removed list: ", apples_to_remove)
    for i in apples_to_remove:
        Apple.existingApples.remove(i)
        del i
        # print("Removed Apples, here is existingApples: ", Apple.existingApples)
        # print("Removed Apples, here is apples_to_remove: ", apples_to_remove)
    # APPLE DRAWING
    for i in Apple.existingApples:
      if i.yPos >= annotated_image.shape[0]:
        continue
      else:
        i.roi(annotated_image)

    # If window closed, exit (This has to be before showing image, I'm not sure why)
    if cv2.getWindowProperty('Catch Apples', cv2.WND_PROP_VISIBLE) < 1:
        break
    # Show image
    cv2.imshow("Catch Apples", annotated_image)


    cv2.waitKey(1) #needed to render image?

    if keyboard.is_pressed("esc"): # break if esc key pressed during playback
        break   

cap.release()
cv2.destroyAllWindows()
