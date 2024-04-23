from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import os
import requests
from tqdm import tqdm
import numpy as np
import pygame

def eye_ratio(eye):
    # Define the eye landmarks
    left_eye_top = eye[1]
    left_eye_bottom = eye[5]
    right_eye_top = eye[2]
    right_eye_bottom = eye[4]
    eye_left_corner = eye[0]
    eye_right_corner = eye[3]

    vertical_dist_1 = dist.euclidean(left_eye_top, left_eye_bottom)
    vertical_dist_2 = dist.euclidean(right_eye_top, right_eye_bottom)

    horizontal_dist = dist.euclidean(eye_left_corner, eye_right_corner)

    ear_numerator = vertical_dist_1 + vertical_dist_2
    ear_denominator = 2.0 * horizontal_dist
    ear = ear_numerator / ear_denominator

    return ear

def mouth_ratio(mouth):
    # Define the points of the mouth
    upper_lip_center = mouth[3]
    lower_lip_center = mouth[9]
    mouth_left_corner = mouth[0]
    mouth_right_corner = mouth[6]

    vertical_center_distance = dist.euclidean(upper_lip_center, lower_lip_center)
    horizontal_distance = dist.euclidean(mouth_left_corner, mouth_right_corner)

    mratio = vertical_center_distance / horizontal_distance

    return mratio

def model_download(url, save_path):
    # Make a request to download the file
    response = requests.get(url, stream=True)
    size = int(response.headers.get('content-length', 0))
    chunk_size = 1024

    progress = tqdm(total=size, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as file:
        for content in response.iter_content(chunk_size):
            progress.update(len(content))
            file.write(content)
    progress.close()

path = "shape_predictor_68_face_landmarks.dat"

# Check if the model exists, otherwise download it
if not os.path.exists(path):
    print("Downloading the shape predictor model...")
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    model_download(url, path + ".bz2")

    # Extract the downloaded file
    os.system(f"bzip2 -d {path}.bz2")
    print("Download complete.")

# Initialize dlib face detector and shape predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(path)

# Facial landmarks for the left eye, right eye, mouth, and nose
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["nose"]

cap = cv2.VideoCapture(0)
flag = 0

# Initialize pygame for sound
pygame.init()
beep_sound = pygame.mixer.Sound("beep-01.wav")
view = "Center"

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Detect eyes and calculate eye aspect ratio
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_ratio(leftEye)
        rightEAR = eye_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Detect mouth and calculate mouth aspect ratio
        mouth = shape[mStart:mEnd]
        mar = mouth_ratio(mouth)

        # Detect nose for rotation detection
        nose = shape[nStart:nEnd]
        nose_tip = nose[3]  # Approximate index for the nose tip in the 68 landmarks model

        # Calculate rotation indicators
        eyes_center = ((leftEye[0] + rightEye[3]) / 2.0)
        nose_horizontal_dist = abs(nose_tip[0] - eyes_center[0])

        eyes_vertical_center = (leftEye[1][1] + rightEye[4][1]) / 2.0
        nose_vertical_dist = abs(nose_tip[1] - eyes_vertical_center)

        horizontal_rotation_threshold = 10
        vertical_rotation_threshold = 10

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
            view = "Center"
        else:
            view = "not Center"
            cv2.putText(frame, "FACE ROTATION ALERT", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord('w'):
        print("Exiting the video stream.")
        break

cap.release()
cv2.destroyAllWindows()
print("Video stream stopped and windows closed.")