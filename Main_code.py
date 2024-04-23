#importing modules
from scipy.spatial import distance as dist
import cv2
import os
from imutils import face_utils
import imutils
import dlib
import requests
import numpy as np
import pygame

#funtion for calculate eye aspect ratio
def ratio_of_eye(eye):
    # Define the eye landmarks
    left_eye_top = eye[1]
    left_eye_bottom = eye[5]
    right_eye_top = eye[2]
    right_eye_bottom = eye[4]
    eye_left_corner = eye[0]
    eye_right_corner = eye[3]

    #calculate euclidean distance for each eye between top and bottom
    vertical_dist_1 = dist.euclidean(left_eye_top, left_eye_bottom)
    vertical_dist_2 = dist.euclidean(right_eye_top, right_eye_bottom)
    horizontal_dist = dist.euclidean(eye_left_corner, eye_right_corner)

    ear_numerator = vertical_dist_1 + vertical_dist_2
    ear_denominator = 2.0 * horizontal_dist
    ear = ear_numerator / ear_denominator

    return ear

#function to calculate mouth aspect ratio
def ratio_of_mouth(mouth):
    # Define the points of the mouth
    upper_lip_center = mouth[3]
    lower_lip_center = mouth[9]
    mouth_left_corner = mouth[0]
    mouth_right_corner = mouth[6]

    #calculate euclidean distance
    vertical_center_distance = dist.euclidean(upper_lip_center, lower_lip_center)
    horizontal_distance = dist.euclidean(mouth_left_corner, mouth_right_corner)

    mratio = vertical_center_distance / horizontal_distance

    return mratio

#function to download the model
def model_download(url, save_path):
    # Make a request to download the file
    response = requests.get(url, stream = True)
    size_of_chunk = 1024

    with open(save_path, 'wb') as file:
        for content in response.iter_content(size_of_chunk):
            file.write(content)

path = "shape_predictor_68_face_landmarks.dat"

# check if the model exists if not then download the model
if not os.path.exists(path):
    print("---Downloading the model---")
    link = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    model_download(link, path + ".bz2")
    
    # Extract the downloaded file
    os.system(f"bzip2 -d {path}.bz2")
    print("Download completed.")

# initialize dlib face detector and shape predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(path)

# landmarks of left eye, right eye, mouth, and nose
(left_Start, left_End) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(right_Start, right_End) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mouth_Start, mouth_End) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
(nose_Start, nose_End) = face_utils.FACIAL_LANDMARKS_68_IDXS["nose"]

video = cv2.VideoCapture(0)
flag = 0

# Initialize pygame for sound
pygame.init()
#get the beep sound file
beep_sound = pygame.mixer.Sound("beep-01.wav")
view = "Center"

while(True):

    ret, frame = video.read()
    #resize frame
    frame = imutils.resize(frame, width = 450)
    #color to gray
    to_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detectedFaces = detect(to_gray, 0)
    
    for detectedFace in detectedFaces:
        shape = predict(to_gray, detectedFace)
        shape = face_utils.shape_to_np(shape)
        
        left_Eye = shape[left_Start:left_End]
        right_Eye = shape[right_Start:right_End]
        left_EAR = ratio_of_eye(left_Eye)
        right_EAR = ratio_of_eye(right_Eye)
        ear = (left_EAR + right_EAR) / 2.0
        
        mouth = shape[mouth_Start:mouth_End]
        mar = ratio_of_mouth(mouth)
        
        # Detect nose for rotation detection
        nose = shape[nose_Start:nose_End]
        nose_tip = nose[3]  # Approximate index for the nose tip in the 68 landmarks model
        
        # Calculate rotation indicators
        eyes_center = ((left_Eye[0] + right_Eye[3]) / 2.0)
        nose_horizontal_dist = abs(nose_tip[0] - eyes_center[0])
        
        eyes_vertical_center = (left_Eye[1][1] + right_Eye[4][1]) / 2.0
        nose_vertical_dist = abs(nose_tip[1] - eyes_vertical_center)
        
        horizontal_rotation_threshold = 10
        vertical_rotation_threshold = 10

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Check for eye alert
        if ear < 0.25 and view == "Center":
            flag += 1
            if flag >= 20:
                cv2.putText(frame, "EYE ALERT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                beep_sound.play()
        else:
            flag = 0
        
        # Check for yawn alert
        if mar > 0.6 and view == "Center":
            cv2.putText(frame, "YAWN ALERT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            beep_sound.play()

        if nose_horizontal_dist < horizontal_rotation_threshold:
            #make view is center
            view = "Center"
        else:
            #make view is not center
            view = "not Center"
            cv2.putText(frame, "FACE ROTATION ALERT", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Facial expression detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
        
        # Check if predictions is a list and is not empty
        if isinstance(predictions, list) and len(predictions) > 0:
            for prediction in predictions:
                dominant_emotion = prediction.get("dominant_emotion", "Unknown")
                
                # check if emotion is happy else if emotion is neutral
                if dominant_emotion == "happy":
                    cv2.putText(frame, "Smiling", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
                elif dominant_emotion == "neutral":
                    cv2.putText(frame, f"{dominant_emotion}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        
        
    #display the frame
    cv2.imshow("Frame", frame)
    #wait for a key entry
    stop = cv2.waitKey(1)
    #stop video stream if pressed 'E'
    if stop == ord('E'):
        print("Exiting the video stream.")
        break

#release video capturing
video.release()

#destroy all windows
cv2.destroyAllWindows()
print("Video stream stopped and windows closed.")
