from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import os
import requests
from tqdm import tqdm
import pygame

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[3], mouth[9])
    B = distance.euclidean(mouth[2], mouth[10])
    C = distance.euclidean(mouth[4], mouth[8])
    D = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B + C) / (3.0 * D)
    return mar

def download_model(url, save_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

model_path = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(model_path):
    print("Downloading the shape predictor model...")
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    download_model(url, model_path + ".bz2")
    os.system(f"bzip2 -d {model_path}.bz2")
    print("Download complete.")

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(model_path)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["nose"]

cap = cv2.VideoCapture(0)
flag = 0

pygame.init()
beep_sound = pygame.mixer.Sound("beep-01.wav")

nose_point = (0, 0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)
        
        if ear < 0.25:
            flag += 1
            if flag >= 20:
                cv2.putText(frame, " EYE ALERT ", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                beep_sound.play()
        else:
            flag = 0
        
        if mar > 0.5:
            cv2.putText(frame, " YAWN ALERT ", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            beep_sound.play()

        nose = shape[nStart:nEnd]
        nose_center = nose[nose.shape[0] // 2]

        if nose_point[0] == 0 and nose_point[1] == 0:
            nose_point = nose_center
        else:
            direction = ""
            if nose_center[0] - nose_point[0] > 40:
                direction = "Head moved to RIGHT"
            elif nose_center[0] - nose_point[0] < -40:
                direction = "Head moved to LEFT"

            if nose_center[1] - nose_point[1] > 40:
                direction = "Head moved UP"
            elif nose_center[1] - nose_point[1] < -40:
                direction = "Head moved DOWN"

            if direction != "":
                cv2.putText(frame, direction, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                nose_point = nose_center

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
