import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Setup and Constants ---

# MediaPipe initializations
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, # We will focus on one dominant hand for data collection
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['jab', 'hook', 'uppercut'])

# Thirty videos worth of data
num_sequences = 50

# Videos are going to be 20 frames in length
sequence_length = 20

# --- Folder Setup ---
# This block creates the folders where your training data will be stored.
# Example hierarchy: MP_Data/jab/0/..., MP_Data/jab/1/..., etc.
for action in actions: 
    for sequence in range(num_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# --- Helper Functions ---

def extract_keypoints(pose_results, hand_results):
    """
    Extracts the landmark data and flattens it into a single numpy array.
    """
    # Extract pose landmarks, or a zero array if not detected
    pose_landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in pose_results.pose_landmarks.landmark]).flatten() if pose_results.pose_landmarks else np.zeros(33*4)
    
    # Extract hand landmarks, or a zero array if not detected
    hand_landmarks = np.array([[res.x, res.y, res.z] for res in hand_results.multi_hand_landmarks[0].landmark]).flatten() if hand_results.multi_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose_landmarks, hand_landmarks])

# --- Main Application Logic ---

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    # Wait for user to be ready
    cv2.waitKey(2000)

    # Loop through actions
    for action in actions:
        # Loop through sequences (videos)
        for sequence in range(num_sequences):
            # Loop through video length (sequence length)
            for frame_num in range(sequence_length + 5): # Add some buffer frames
                success, frame = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                
                # Make detections
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                
                pose_result = pose.process(rgb_frame)
                hands_result = hands.process(rgb_frame)
                
                rgb_frame.flags.writeable = True
                frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                
                # Draw landmarks
                if pose_result.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if hands_result.multi_hand_landmarks:
                    for hand_landmarks in hands_result.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # --- Data Collection Logic ---
                if frame_num < 5: 
                    # Display "GET READY" message for the first 5 frames
                    cv2.putText(frame, 'GET READY...', (120, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                    cv2.putText(frame, f'Collecting frames for {action} | Sequence Number {sequence}', (15, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Boxing Analysis - Data Collection', frame)
                    cv2.waitKey(500) # Wait half a second between ready frames
                else: 
                    # Display "RECORDING" message
                    cv2.putText(frame, 'RECORDING...', (120, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, f'Collecting frames for {action} | Sequence Number {sequence}', (15, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Boxing Analysis - Data Collection', frame)

                    # Export keypoints
                    keypoints = extract_keypoints(pose_result, hands_result)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num - 5))
                    np.save(npy_path, keypoints)
                
                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()

if __name__ == '__main__':
    main()