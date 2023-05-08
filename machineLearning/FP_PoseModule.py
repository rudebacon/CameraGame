import cv2
import mediapipe as mp




class poseDetector():
 
    def __init__(self,
               mode=False,
               complexity=1,
               smooth=True,
               en_seg=False,
               sm_seg=True,
               detectCon=0.5,
               trackCon=0.5):
 
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.en_seg = en_seg
        self.sm_seg = sm_seg
        self.detectCon = detectCon
        self.trackCon = trackCon
 
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth,
                                     self.en_seg, self.sm_seg, self.detectCon, self.trackCon)
 
    def findPose(self, img, draw=True): #Processes image and draws lines
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS, self.mpDraw.DrawingSpec(color=(255,255,255), thickness=0, circle_radius=0),
            self.mpDraw.DrawingSpec(color=(255,255,255), thickness=2))
        
        return img


    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                
                # NOTE: We cannot do anything three dimensional unless we get a stereo camera setup.
                
                h, w, c = img.shape
                
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 4, (255, 0, 0), cv2.FILLED)
              
        return self.lmList


    def findLocalWorldLandmarks(self):
        self.localList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_world_landmarks.landmark):
                self.localList.append([id, lm.x, lm.y * -1, lm.z * -1])
        
        return self.localList


        


    # def findAngle(self, img, points, p1, p2, p3, draw=False):
    #     """Finds the angle in the given image between three given points, and draws the points and lines between them.

    #     Parameters:

    #     img - numpy.array
    #     The camera feed being shown by the OpenCV code.

    #     points - list
    #     A list containing the IDs of the points being used. "O" stands for the origin, which is between the two hips.
    #     Used for drawing the lines on the person's body.

    #     p1, p2, p3 - list/tuple
    #     Each contains the x, y, and z coordinates of the desired point in meters relative to the midpoint of the hips.

    #     draw - boolean
    #     Whether you want to draw the lines of the angle on the image or not.
    #     """

    #     # Getting (x,y) coordinates of two vectors (p2 --> p1, p2 --> p3)
    #     xa, ya, za = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
    #     xb, yb, zb = p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]

    #     # Calculate the Angle
    #     angle = math.degrees(math.acos((xa * xb + ya * yb + za * zb) /
    #                                    (math.sqrt(xa * xa + ya * ya + za * za) *
    #                                     math.sqrt(xb * xb + yb * yb + zb * zb))))

    #     if angle < 0:
    #         angle += 360

    #     if angle > 180:
    #         angle = 360 - angle

    #     # Draw
    #     if draw:

    #         # Getting the pixel values of the points
    #         x1 = self.lmList[points[0]][1]
    #         y1 = self.lmList[points[0]][2]

    #         if points[1] == "O":
    #             x2 = int((self.lmList[23][1] + self.lmList[24][1]) / 2)
    #             y2 = int((self.lmList[23][2] + self.lmList[24][2]) / 2)
    #         else:
    #             x2 = self.lmList[points[1]][1]
    #             y2 = self.lmList[points[1]][2]

    #         x3 = self.lmList[points[2]][1]
    #         y3 = self.lmList[points[2]][2]

    #         # Drawing the lines
    #         cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
    #         cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
    #         '''cv2.circle(img, (x1, y1), 4, (255, 200, 0), cv2.FILLED)
    #         cv2.circle(img, (x1, y1), 7, (255, 200, 0), 2)
    #         cv2.circle(img, (x2, y2), 4, (255, 200, 0), cv2.FILLED)
    #         cv2.circle(img, (x2, y2), 7, (255, 200, 0), 2)
    #         cv2.circle(img, (x3, y3), 4, (255, 200, 0), cv2.FILLED)
    #         cv2.circle(img, (x3, y3), 7, (255, 200, 0), 2)'''
    #         cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
    #                     cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    #     return angle

    # def findDist(self, img, points, p1, p2, draw=False):
    #     """Finds the angle in the given image between three given points, and draws the points and lines between them.

    #     Parameters:

    #     img - numpy.array
    #     The camera feed being shown by the OpenCV code.

    #     points - list
    #     A list containing the IDs of the points being used. "O" stands for the origin, which is between the two hips.
    #     Used for drawing the lines on the person's body.

    #     p1, p2 - list/tuple
    #     Each contains the x, y, and z coordinates of the desired point in meters relative to the midpoint of the hips.

    #     draw - boolean
    #     Whether you want to draw the lines of the angle on the image or not.
    #     """

    #     # Get the landmarks
    #     xa, ya, za = list(p1)
    #     xb, yb, zb = list(p2)

    #     print("a =", xa, ya, za)
    #     print("b =", xb, yb, zb)

    #     # Calculate the Distance
    #     dist = math.sqrt((xa - xb) * (xa - xb) + (ya - yb) * (ya - yb) + (za - zb) * (za - zb))

    #     print("dist =", dist)

    #     # Draw
    #     if draw:
    #         # Getting the pixel values of the points
    #         x1 = self.lmList[points[0]][1]
    #         y1 = self.lmList[points[0]][2]

    #         x2 = self.lmList[points[1]][1]
    #         y2 = self.lmList[points[1]][2]

    #         cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
    #         cv2.putText(img, str(round(dist, 3)), (int((x1 + x2) / 2), int((y1 + y2) / 2)),
    #                     cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    #     return dist
    
    

        

def main():
    # cap = cv2.VideoCapture(1)
    # pTime = 0
    # detector = poseDetector()
    # while True:
    #     success, img = cap.read()
    #     img = detector.findPose(img)
    #     lmList = detector.findPosition(img, draw=False)
    #     if len(lmList) != 0:
    #         print(lmList[0])
    #         cv2.circle(img, (lmList[0][1], lmList[0][2]), 15, (0, 0, 255), cv2.FILLED)
 
    #     cTime = time.time()
    #     fps = 1 / (cTime - pTime)
    #     pTime = cTime
 
    #     cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
    #                 (255, 0, 0), 3)
 
    #     cv2.imshow("Image", img)
    #     cv2.waitKey(1)
    pass
 
 
if __name__ == "__main__":
    main()