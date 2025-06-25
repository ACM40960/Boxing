import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf # Changed from pickle
import time
import math
import os
import pygame
import csv
from datetime import datetime

# --- 1. Initialization (Audio, Model, MediaPipe) ---
print("Initializing components...")
pygame.mixer.init()

def load_sound(name):
    path = os.path.join('audio', f'{name}.wav')
    if not os.path.exists(path): path = os.path.join('audio', f'{name}.mp3')
    if not os.path.exists(path): return None
    return pygame.mixer.Sound(path)

sounds = {'jab': load_sound('jab'), 'hook': load_sound('hook'), 'uppercut': load_sound('uppercut'), 'hit': load_sound('hit'), 'miss': load_sound('miss')}

# ** NEW: Load the Keras LSTM model **
print("Loading LSTM model 'action_model.keras'...")
MODEL_PATH = 'action_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, max_num_poses=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

actions = np.array(['jab', 'hook', 'uppercut'])
sequence_length = 20

# --- Helper Functions (No changes) ---
def extract_keypoints(pose_results, hand_results):
    pose_landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in pose_results.landmark]).flatten() if pose_results else np.zeros(33*4)
    hand_landmarks = np.array([[res.x, res.y, res.z] for res in hand_results.landmark]).flatten() if hand_results else np.zeros(21*3)
    return np.concatenate([pose_landmarks, hand_landmarks])

def is_fist(hand_landmarks):
    if not hand_landmarks: return False
    finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    pip_joints = [mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]
    fingers_curled = all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y for tip, pip in zip(finger_tips, pip_joints))
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    thumb_tucked = math.hypot(thumb_tip.x - index_mcp.x, thumb_tip.y - index_mcp.y) < 0.1
    return fingers_curled and thumb_tucked

# --- 4. Main Application: boxing_analyzer_lstm.py ---
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # State, counters, and logging list
    session_log = []
    sequence_buffer, last_punch_time, punch_cooldown, last_hand_state = [], 0, 1.0, 'OPEN'
    current_prediction, feedback_text, feedback_timer = "---", "", 0
    hit_counts = {'jab': 0, 'hook': 0, 'uppercut': 0}
    miss_count, foul_count = 0, 0
    
    ATTACKER_COLOR, DEFENDER_COLOR, HIT_COLOR, MISS_COLOR, TARGET_COLOR = (255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 100, 255), (255, 255, 0)
    feedback_duration = 1.5

    print("Starting session with LSTM model. Press 'q' to quit and save the log.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        rgb_frame.flags.writeable = False
        pose_results = pose.process(rgb_frame)
        hands_results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        # Player ID and Target Zone logic (same as before)
        # ... (re-using the robust logic from the final script) ...
        attacker_pose, defender_pose, attacker_hand = None, None, None
        if pose_results.pose_landmarks and len(pose_results.pose_landmarks) > 0:
            poses = sorted(pose_results.pose_landmarks, key=lambda p: p.landmark[mp_pose.PoseLandmark.NOSE].x)
            attacker_pose = poses[0]
            if len(poses) > 1: defender_pose = poses[1]
        
        if hands_results.multi_hand_landmarks and attacker_pose:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                if abs(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x - attacker_pose.landmark[mp_pose.PoseLandmark.NOSE].x) < 0.3:
                    attacker_hand = hand_landmarks; break
        
        head_target, body_target = None, None
        if defender_pose:
            nose = defender_pose.landmark[mp_pose.PoseLandmark.NOSE]; l_ear = defender_pose.landmark[mp_pose.PoseLandmark.LEFT_EAR]; r_ear = defender_pose.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
            head_target = (int(l_ear.x * w), int(nose.y * h) - int(abs(l_ear.x-r_ear.x)*w)//2, int(r_ear.x * w), int(nose.y * h) + int(abs(l_ear.x-r_ear.x)*w)//2)
            l_shoulder = defender_pose.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]; r_shoulder = defender_pose.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]; l_hip = defender_pose.landmark[mp_pose.PoseLandmark.LEFT_HIP]; r_hip = defender_pose.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            body_target = (int(l_shoulder.x * w), int(((l_shoulder.y+r_shoulder.y)/2)*h), int(r_shoulder.x * w), int(((l_hip.y+r_hip.y)/2)*h))
            cv2.rectangle(frame, (head_target[0], head_target[1]), (head_target[2], head_target[3]), TARGET_COLOR, 2)
            cv2.rectangle(frame, (body_target[0], body_target[1]), (body_target[2], body_target[3]), TARGET_COLOR, 2)


        # ** UPDATED: Punch Classification with LSTM Model **
        if attacker_pose:
            sequence_buffer.append(extract_keypoints(attacker_pose, attacker_hand))
            sequence_buffer = sequence_buffer[-sequence_length:]
            
            current_time = time.time()
            current_hand_state = 'FIST' if is_fist(attacker_hand) else 'OPEN'
            
            if current_hand_state == 'FIST' and last_hand_state == 'OPEN' and (current_time - last_punch_time > punch_cooldown):
                if len(sequence_buffer) == sequence_length:
                    
                    # ** CHANGE 1: Prepare data for LSTM (3D shape) **
                    # No more flattening. Just add a "batch" dimension.
                    input_data = np.expand_dims(sequence_buffer, axis=0) 
                    
                    # ** CHANGE 2: Get prediction from Keras model **
                    prediction_probs = model.predict(input_data)[0]
                    prediction = np.argmax(prediction_probs)
                    confidence = prediction_probs[prediction]
                    
                    predicted_action = actions[prediction]
                    
                    if confidence > 0.8: # Confidence threshold for the new model
                        # (The rest of the logic for hit/miss/logging is the same)
                        current_prediction = f"{predicted_action.upper()} ({confidence:.2f})"
                        last_punch_time, feedback_timer = current_time, current_time
                        if sounds.get(predicted_action): sounds[predicted_action].play()

                        px, py = (0,0)
                        if attacker_hand: px, py = int(attacker_hand.landmark[mp_hands.HandLandmark.WRIST].x * w), int(attacker_hand.landmark[mp_hands.HandLandmark.WRIST].y * h)
                        hit = (defender_pose and ((head_target and head_target[0]<px<head_target[2] and head_target[1]<py<head_target[3]) or (body_target and body_target[0]<px<body_target[2] and body_target[1]<py<body_target[3])))
                        outcome = "HIT" if hit else "MISS"
                        feedback_text = f"{outcome}!"
                        if hit:
                            hit_counts[predicted_action] += 1
                            if sounds.get('hit'): sounds['hit'].play()
                        else:
                            miss_count += 1
                            if sounds.get('miss'): sounds['miss'].play()
                        
                        session_log.append({'timestamp': datetime.now().isoformat(), 'punch_type': predicted_action, 'confidence': f"{confidence:.4f}", 'outcome': outcome})
            last_hand_state = current_hand_state

        # Visualization and HUD (same as before)
        # ... (re-using final visualization code) ...
        bgr_frame = frame
        if attacker_pose: mp_drawing.draw_landmarks(bgr_frame, attacker_pose, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=ATTACKER_COLOR, thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=ATTACKER_COLOR, thickness=2, circle_radius=2))
        if defender_pose: mp_drawing.draw_landmarks(bgr_frame, defender_pose, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=DEFENDER_COLOR, thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=DEFENDER_COLOR, thickness=2, circle_radius=2))
        if attacker_hand: mp_drawing.draw_landmarks(bgr_frame, attacker_hand, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=ATTACKER_COLOR, thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=ATTACKER_COLOR, thickness=2, circle_radius=2))
        if time.time() - feedback_timer < feedback_duration: cv2.putText(bgr_frame, feedback_text, (int(w/2)-100, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, HIT_COLOR if feedback_text == "HIT!" else MISS_COLOR, 3)
        cv2.rectangle(bgr_frame, (0, 0), (w, 60), (245, 117, 16), -1)
        cv2.putText(bgr_frame, 'LAST PUNCH:', (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(bgr_frame, current_prediction, (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        score_text = f"HITS: J:{hit_counts['jab']} H:{hit_counts['hook']} U:{hit_counts['uppercut']} | MISSES: {miss_count} | FOULS: {foul_count}"
        cv2.putText(bgr_frame, score_text, (w - 750, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        cv2.imshow("Boxing Analyzer - LSTM Version", bgr_frame)
        if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    
    # Save Session Log on Exit (same as before)
    if session_log:
        filename = f"session_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        print(f"Session ended. Saving log to {filename}...")
        headers = ['timestamp', 'punch_type', 'confidence', 'outcome']
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(session_log)
        print("Log saved successfully.")

if __name__ == '__main__':
    main()