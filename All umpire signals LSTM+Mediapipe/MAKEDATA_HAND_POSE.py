import cv2
import mediapipe as mp
import pandas as pd

# Read video from file
video_path = r"c:\PROJECT\Final Year Project\Own dataset\short_run.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize Mediapipe Pose and Hands
mpPose = mp.solutions.pose
mpHands = mp.solutions.hands
pose = mpPose.Pose()
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "test"
no_of_frames = 600

def make_landmark_timestep(pose_results, hands_results):
    c_lm = []
    
    # Extract Pose landmarks
    if pose_results.pose_landmarks:
        for lm in pose_results.pose_landmarks.landmark:
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            c_lm.append(lm.visibility)
    
    # Extract Hand landmarks (for both hands)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                c_lm.append(lm.x)
                c_lm.append(lm.y)
                c_lm.append(lm.z)
    
    return c_lm

def draw_landmarks(mpDraw, pose_results, hands_results, img):
    # Draw Pose landmarks
    if pose_results.pose_landmarks:
        mpDraw.draw_landmarks(img, pose_results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    
    # Draw Hand landmarks
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    return img

while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process Pose and Hand landmarks
    pose_results = pose.process(frameRGB)
    hands_results = hands.process(frameRGB)
    
    # Collect landmark data
    lm = make_landmark_timestep(pose_results, hands_results)
    lm_list.append(lm)
    
    # Draw Pose and Hand landmarks
    frame = draw_landmarks(mpDraw, pose_results, hands_results, frame)

    cv2.imshow("image", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Save landmark data to CSV
df = pd.DataFrame(lm_list)
df.to_csv(label + ".txt", index=False)

cap.release()
cv2.destroyAllWindows()
