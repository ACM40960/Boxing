import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import math
import os
import pygame # Added for audio feedback

# --- 1. Initialize Pygame Mixer and Load Sounds ---
pygame.mixer.init()

# Define a function to safely load sounds
def load_sound(name):
    path = os.path.join('audio', f'{name}.wav') # Prefers .wav but can try .mp3
    if not os.path.exists(path):
        path = os.path.join('audio', f'{name}.mp3')
        if not os.path.exists(path):
            print(f"Warning: Sound file not found for '{name}'")
            return None
    return pygame.mixer.Sound(path)

sounds = {
    'jab': load_sound('jab'),
    'hook': load_sound('hook'),
    'uppercut': load_sound('uppercut'),
    'hit': load_sound('hit'),
    'miss': load_sound('miss')
}

# --- 2. Load the Trained Model ---
MODEL_PATH = 'boxing_model.pkl'
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# --- 3. Setup MediaPipe ---
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, max_num_poses=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

actions = np.array(['jab', 'hook', 'uppercut'])
sequence_length = 20

# --- Helper Functions (No changes from previous version) ---
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

# --- 4. Main Application: sparring_analyzer_v3.py ---
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # State and counters
    sequence_buffer, last_punch_time, punch_cooldown, last_hand_state = [], 0, 1.0, 'OPEN'
    current_prediction, feedback_text, feedback_timer = "---", "", 0
    hit_counts = {'jab': 0, 'hook': 0, 'uppercut': 0}
    miss_count = 0
    foul_count = 0 # Placeholder for foul moves
    
    # Colors and settings
    ATTACKER_COLOR, DEFENDER_COLOR, HIT_COLOR, MISS_COLOR, TARGET_COLOR = (255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 100, 255), (255, 255, 0)
    feedback_duration = 1.5

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
        
        # Player Identification and Target Zone logic (same as v2)
        attacker_pose, defender_pose, attacker_hand = None, None, None
        if pose_results.pose_landmarks and len(pose_results.pose_landmarks) > 0:
            poses = sorted(pose_results.pose_landmarks, key=lambda p: p.landmark[mp_pose.PoseLandmark.NOSE].x)
            attacker_pose = poses[0]
            if len(poses) > 1: defender_pose = poses[1]
        
        if hands_results.multi_hand_landmarks and attacker_pose:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                if abs(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x - attacker_pose.landmark[mp_pose.PoseLandmark.NOSE].x) < 0.3:
                    attacker_hand = hand_landmarks
                    break

        head_target, body_target = None, None
        if defender_pose:
            # (Re-using the same target definition logic from v2)
            nose = defender_pose.landmark[mp_pose.PoseLandmark.NOSE]; l_ear = defender_pose.landmark[mp_pose.PoseLandmark.LEFT_EAR]; r_ear = defender_pose.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
            head_target = (int(l_ear.x * w), int(nose.y * h) - int(abs(l_ear.x - r_ear.x) * w)//2, int(r_ear.x * w), int(nose.y * h) + int(abs(l_ear.x - r_ear.x) * w)//2)
            cv2.rectangle(frame, (head_target[0], head_target[1]), (head_target[2], head_target[3]), TARGET_COLOR, 2)
            l_shoulder = defender_pose.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]; r_shoulder = defender_pose.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]; l_hip = defender_pose.landmark[mp_pose.PoseLandmark.LEFT_HIP]; r_hip = defender_pose.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            body_target = (int(l_shoulder.x * w), int(((l_shoulder.y + r_shoulder.y) / 2) * h), int(r_shoulder.x * w), int(((l_hip.y + r_hip.y) / 2) * h))
            cv2.rectangle(frame, (body_target[0], body_target[1]), (body_target[2], body_target[3]), TARGET_COLOR, 2)

        # Punch Classification and Hit Detection with AUDIO
        if attacker_pose:
            keypoints = extract_keypoints(attacker_pose, attacker_hand)
            sequence_buffer.append(keypoints)
            sequence_buffer = sequence_buffer[-sequence_length:]
            
            current_time = time.time()
            current_hand_state = 'FIST' if is_fist(attacker_hand) else 'OPEN'
            
            if current_hand_state == 'FIST' and last_hand_state == 'OPEN' and (current_time - last_punch_time > punch_cooldown):
                if len(sequence_buffer) == sequence_length:
                    input_data = np.expand_dims(sequence_buffer, axis=0).reshape(1, -1)
                    prediction = model.predict(input_data)
                    confidence = model.predict_proba(input_data)
                    predicted_action = actions[prediction[0]]
                    
                    if np.max(confidence) > 0.75:
                        current_prediction = f"{predicted_action.upper()} ({np.max(confidence):.2f})"
                        last_punch_time, feedback_timer = current_time, current_time
                        if sounds[predicted_action]: sounds[predicted_action].play() # ** PLAY PUNCH SOUND **

                        # HIT or MISS?
                        punch_coord = attacker_hand.landmark[mp_hands.HandLandmark.WRIST]
                        px, py = int(punch_coord.x * w), int(punch_coord.y * h)
                        hit = (defender_pose and 
                               ((head_target and head_target[0] < px < head_target[2] and head_target[1] < py < head_target[3]) or 
                                (body_target and body_target[0] < px < body_target[2] and body_target[1] < py < body_target[3])))
                        
                        if hit:
                            feedback_text = "HIT!"
                            hit_counts[predicted_action] += 1
                            if sounds['hit']: sounds['hit'].play() # ** PLAY HIT SOUND **
                        else:
                            feedback_text = "MISS!"
                            miss_count += 1
                            if sounds['miss']: sounds['miss'].play() # ** PLAY MISS SOUND **
            last_hand_state = current_hand_state

        # Visualization (same as v2, with updated HUD)
        bgr_frame = frame
        # Draw skeletons, targets, etc...
        if attacker_pose: mp_drawing.draw_landmarks(bgr_frame, attacker_pose, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=ATTACKER_COLOR, thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=ATTACKER_COLOR, thickness=2, circle_radius=2))
        if defender_pose: mp_drawing.draw_landmarks(bgr_frame, defender_pose, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=DEFENDER_COLOR, thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=DEFENDER_COLOR, thickness=2, circle_radius=2))
        if attacker_hand: mp_drawing.draw_landmarks(bgr_frame, attacker_hand, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=ATTACKER_COLOR, thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=ATTACKER_COLOR, thickness=2, circle_radius=2))

        if time.time() - feedback_timer < feedback_duration:
            cv2.putText(bgr_frame, feedback_text, (int(w/2)-100, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, HIT_COLOR if feedback_text == "HIT!" else MISS_COLOR, 3)

        # Updated Scoreboard HUD
        cv2.rectangle(bgr_frame, (0, 0), (w, 60), (245, 117, 16), -1)
        cv2.putText(bgr_frame, 'LAST PUNCH:', (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(bgr_frame, current_prediction, (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        score_text = f"HITS: J:{hit_counts['jab']} H:{hit_counts['hook']} U:{hit_counts['uppercut']} | MISSES: {miss_count} | FOULS: {foul_count}"
        cv2.putText(bgr_frame, score_text, (w - 750, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Sparring Analyzer v3 - with Audio", bgr_frame)
        if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()