import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import math

# --- 1. Load the Trained Model ---
MODEL_PATH = 'boxing_model.pkl'
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# --- 2. Setup MediaPipe for Two Players ---
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, max_num_poses=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

actions = np.array(['jab', 'hook', 'uppercut'])
sequence_length = 20

# --- 3. Helper Functions ---
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

# --- 4. Main Application: sparring_analyzer_v2.py ---
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # State and counters
    sequence_buffer = []
    last_punch_time = 0
    punch_cooldown = 1.0
    last_hand_state = 'OPEN'
    current_prediction = "---"
    
    hit_counts = {'jab': 0, 'hook': 0, 'uppercut': 0}
    miss_count = 0
    
    feedback_text = ""
    feedback_timer = 0
    feedback_duration = 1.5 # seconds

    # Colors
    ATTACKER_COLOR = (255, 0, 0) # Blue
    DEFENDER_COLOR = (0, 0, 255) # Red
    HIT_COLOR = (0, 255, 0) # Green
    MISS_COLOR = (0, 100, 255) # Orange/Red
    TARGET_COLOR = (255, 255, 0) # Cyan

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
        
        # Player Identification
        attacker_pose, defender_pose, attacker_hand = None, None, None
        if pose_results.pose_landmarks and len(pose_results.pose_landmarks) > 0:
            poses = sorted(pose_results.pose_landmarks, key=lambda p: p.landmark[mp_pose.PoseLandmark.NOSE].x)
            attacker_pose = poses[0]
            if len(poses) > 1:
                defender_pose = poses[1]
        
        if hands_results.multi_hand_landmarks and attacker_pose:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                attacker_nose_x = attacker_pose.landmark[mp_pose.PoseLandmark.NOSE].x
                if abs(wrist_x - attacker_nose_x) < 0.3:
                    attacker_hand = hand_landmarks
                    break

        # Define and draw defender's target zones
        head_target, body_target = None, None
        if defender_pose:
            # Head target
            nose = defender_pose.landmark[mp_pose.PoseLandmark.NOSE]
            l_ear = defender_pose.landmark[mp_pose.PoseLandmark.LEFT_EAR]
            r_ear = defender_pose.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
            head_x1 = int(l_ear.x * w)
            head_y1 = int(nose.y * h) - (int(abs(l_ear.x - r_ear.x) * w) // 2)
            head_x2 = int(r_ear.x * w)
            head_y2 = int(nose.y * h) + (int(abs(l_ear.x - r_ear.x) * w) // 2)
            head_target = (head_x1, head_y1, head_x2, head_y2)
            cv2.rectangle(frame, (head_x1, head_y1), (head_x2, head_y2), TARGET_COLOR, 2)
            cv2.putText(frame, 'HEAD', (head_x1, head_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TARGET_COLOR, 2)

            # Body target
            l_shoulder = defender_pose.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shoulder = defender_pose.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hip = defender_pose.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = defender_pose.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            body_x1 = int(l_shoulder.x * w)
            body_y1 = int(((l_shoulder.y + r_shoulder.y) / 2) * h)
            body_x2 = int(r_shoulder.x * w)
            body_y2 = int(((l_hip.y + r_hip.y) / 2) * h)
            body_target = (body_x1, body_y1, body_x2, body_y2)
            cv2.rectangle(frame, (body_x1, body_y1), (body_x2, body_y2), TARGET_COLOR, 2)
            cv2.putText(frame, 'BODY', (body_x1, body_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TARGET_COLOR, 2)


        # Punch Classification and Hit Detection
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
                        last_punch_time = current_time
                        feedback_timer = current_time

                        # HIT or MISS?
                        if defender_pose and attacker_hand:
                            punch_coord = attacker_hand.landmark[mp_hands.HandLandmark.WRIST]
                            px, py = int(punch_coord.x * w), int(punch_coord.y * h)
                            
                            hit = False
                            if head_target and (head_target[0] < px < head_target[2]) and (head_target[1] < py < head_target[3]):
                                hit = True
                            if body_target and (body_target[0] < px < body_target[2]) and (body_target[1] < py < body_target[3]):
                                hit = True
                            
                            if hit:
                                feedback_text = "HIT!"
                                hit_counts[predicted_action] += 1
                            else:
                                feedback_text = "MISS!"
                                miss_count += 1
                        else:
                            feedback_text = "MISS!"
                            miss_count += 1
            last_hand_state = current_hand_state

        # Visualization
        bgr_frame = frame
        if attacker_pose:
            mp_drawing.draw_landmarks(bgr_frame, attacker_pose, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=ATTACKER_COLOR, thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=ATTACKER_COLOR, thickness=2, circle_radius=2))
            nose_coords = attacker_pose.landmark[mp_pose.PoseLandmark.NOSE]
            cv2.putText(bgr_frame, 'ATTACKER', (int(nose_coords.x * w) - 50, int(nose_coords.y * h) - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, ATTACKER_COLOR, 2)
            if attacker_hand:
                mp_drawing.draw_landmarks(bgr_frame, attacker_hand, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=ATTACKER_COLOR, thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=ATTACKER_COLOR, thickness=2, circle_radius=2))
        if defender_pose:
            mp_drawing.draw_landmarks(bgr_frame, defender_pose, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=DEFENDER_COLOR, thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=DEFENDER_COLOR, thickness=2, circle_radius=2))
            nose_coords = defender_pose.landmark[mp_pose.PoseLandmark.NOSE]
            cv2.putText(bgr_frame, 'DEFENDER', (int(nose_coords.x * w) - 50, int(nose_coords.y * h) - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, DEFENDER_COLOR, 2)
        
        # Display feedback
        if time.time() - feedback_timer < feedback_duration:
            color = HIT_COLOR if feedback_text == "HIT!" else MISS_COLOR
            cv2.putText(bgr_frame, feedback_text, (int(w/2)-100, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 3, cv2.LINE_AA)

        # Scoreboard and HUD
        cv2.rectangle(bgr_frame, (0, 0), (w, 60), (245, 117, 16), -1)
        cv2.putText(bgr_frame, 'LAST PUNCH:', (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr_frame, current_prediction, (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        score_text = f"HITS: J:{hit_counts['jab']} H:{hit_counts['hook']} U:{hit_counts['uppercut']} | MISSES: {miss_count}"
        cv2.putText(bgr_frame, score_text, (w - 600, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Sparring Analyzer v2", bgr_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()

if __name__ == '__main__':
    main()