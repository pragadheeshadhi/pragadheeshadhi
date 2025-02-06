#INFERENCE CODE FOR CRICKET ACTION DETECTION
import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf

label = "LOADING..."
n_time_steps = 10
lm_list = []

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model(r"MiAI_Human_Activity_Recognition-main/MiAI_Human_Activity_Recognition-main/Cricket_model.h5")
video_path=rf"c:\Users\praga\Downloads\deadball and four.mp4"
cap = cv2.VideoCapture(video_path)

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    return img

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, label, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return img

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    predicted_class = np.argmax(results)
    label = signals[predicted_class]

i = 0
warmup_frames = 60

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    i += 1
    
    if i > warmup_frames and results.pose_landmarks:
        c_lm = make_landmark_timestep(results)
        lm_list.append(c_lm)
        if len(lm_list) == n_time_steps:
            threading.Thread(target=detect, args=(model, lm_list,)).start()
            lm_list = []
        img = draw_landmark_on_image(mpDraw, results, img)
    
    img = draw_class_on_image(label, img)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
