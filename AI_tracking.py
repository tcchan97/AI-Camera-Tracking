import mediapipe as mp
import cv2 

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


color = mp_drawing.DrawingSpec(color=(255,255,255),thickness=1,circle_radius=1)

# cap = cv2.VideoCapture(-1)

# while cap.isOpened():
#   ret, frame = cap.read()
#   cv2.imshow("Holistic Model Detection", frame)

#   #if cv2.waitKey(10) && (0xFF === ord('q')):
#   #  break

# cap.release()
# cv2.destroyAllWindows()

cv2.namedWindow("Holistic Model Test")
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
  while( cap.isOpened() ) :
      ret,img = cap.read()
      #Recolor Feed
      image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      #Detections
      results = holistic.process(image)
      #print(results)

      #Recolor Feed back to BGR for rendering
      image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

      #Draw Face Landmarks
      mp_drawing.draw_landmarks(image,results.face_landmarks, mp_holistic.FACE_CONNECTIONS,color)

      #Draw Right Hand Landmarks
      mp_drawing.draw_landmarks(image,results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,color)

      #Draw Left Hand Landmarks
      mp_drawing.draw_landmarks(image,results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,color)

      #Draw Pose Landmarks
      #mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)


      cv2.imshow("Holistic Model Test",image)
      k = cv2.waitKey(1)
      if k == 27:
          break