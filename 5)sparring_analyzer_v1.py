import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import math

# --- 1. Load the Trained Model ---
MODEL_PATH = 'boxing_model.pkl'
print("Loading the trained model...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
print("Model loaded successfully.")

# --- 2. Setup MediaPipe for Two Players ---
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Increased max_num_hands and max_num_poses to 2
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    max_num_poses=2, # Key change for two-player tracking
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

actions = np.array(['jab', 'hook', 'uppercut'])
sequence_length = 20

# --- 3. Helper Functions (from previous steps) ---

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

# --- 4. Main Application Logic ---
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Prediction variables
    sequence_buffer = []
    last_punch_time = 0
    punch_cooldown = 1.0
    last_hand_state = 'OPEN'
    current_prediction = "---"

    # Drawing colors
    ATTACKER_COLOR = (255, 0, 0) # Blue
    DEFENDER_COLOR = (0, 0, 255) # Red

    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        rgb_frame.flags.writeable = False
        pose_results = pose.process(rgb_frame)
        hands_results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # --- Player Identification ---
        attacker_pose = None
        defender_pose = None
        attacker_hand = None

        if pose_results.pose_landmarks:
            # Simple logic: Player on the left is attacker, player on the right is defender
            if len(pose_results.pose_landmarks) == 2:
                poses = pose_results.pose_landmarks
                # Using nose x-coordinate to determine left/right
                if poses[0].landmark[mp_pose.PoseLandmark.NOSE].x < poses[1].landmark[mp_pose.PoseLandmark.NOSE].x:
                    attacker_pose = poses[0]
                    defender_pose = poses[1]
                else:
                    attacker_pose = poses[1]
                    defender_pose = poses[0]
            elif len(pose_results.pose_landmarks) == 1:
                # If only one person, they are the attacker
                attacker_pose = pose_results.pose_landmarks[0]

        # Associate hands with the attacker
        if hands_results.multi_hand_landmarks and attacker_pose:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                # A simple check: if hand's wrist is close to attacker's wrist
                wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                attacker_wrist_x = (attacker_pose.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x + attacker_pose.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x) / 2
                if abs(wrist_x - attacker_wrist_x) < 0.2: # Threshold to associate hand
                    attacker_hand = hand_landmarks
                    break # Assume attacker uses one hand for punching at a time for simplicity

        # --- Punch Classification Logic (for Attacker only) ---
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
                    if np.max(confidence) > 0.7:
                        current_prediction = f"{predicted_action.upper()} ({np.max(confidence):.2f})"
                        last_punch_time = current_time
            last_hand_state = current_hand_state
            
        # --- Visualization ---
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Draw Attacker
        if attacker_pose:
            mp_drawing.draw_landmarks(bgr_frame, attacker_pose, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=ATTACKER_COLOR, thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=ATTACKER_COLOR, thickness=2, circle_radius=2))
            if attacker_hand:
                mp_drawing.draw_landmarks(bgr_frame, attacker_hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=ATTACKER_COLOR, thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=ATTACKER_COLOR, thickness=2, circle_radius=2))
            # Put label
            nose_coords = attacker_pose.landmark[mp_pose.PoseLandmark.NOSE]
            h, w, _ = bgr_frame.shape
            cv2.putText(bgr_frame, 'ATTACKER', (int(nose_coords.x * w) - 50, int(nose_coords.y * h) - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, ATTACKER_COLOR, 2, cv2.LINE_AA)
                        
        # Draw Defender
        if defender_pose:
            mp_drawing.draw_landmarks(bgr_frame, defender_pose, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=DEFENDER_COLOR, thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=DEFENDER_COLOR, thickness=2, circle_radius=2))
            nose_coords = defender_pose.landmark[mp_pose.PoseLandmark.NOSE]
            h, w, _ = bgr_frame.shape
            cv2.putText(bgr_frame, 'DEFENDER', (int(nose_coords.x * w) - 50, int(nose_coords.y * h) - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, DEFENDER_COLOR, 2, cv2.LINE_AA)

        # Display Prediction HUD
        cv2.rectangle(bgr_frame, (0, 0), (640, 60), (245, 117, 16), -1)
        cv2.putText(bgr_frame, 'LAST PUNCH', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr_frame, current_prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Two-Player Boxing Analysis", bgr_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()

if __name__ == '__main__':
    main()