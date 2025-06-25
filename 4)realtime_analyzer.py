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

# --- 2. Setup MediaPipe and Constants ---
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

actions = np.array(['jab', 'hook', 'uppercut'])
sequence_length = 20 # Must match the length used during training

# --- 3. Helper Functions ---

def extract_keypoints(pose_results, hand_results):
    """ Extracts the landmark data and flattens it into a single numpy array. """
    pose_landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in pose_results.pose_landmarks.landmark]).flatten() if pose_results.pose_landmarks else np.zeros(33*4)
    hand_landmarks = np.array([[res.x, res.y, res.z] for res in hand_results.multi_hand_landmarks[0].landmark]).flatten() if hand_results.multi_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose_landmarks, hand_landmarks])

def is_fist(hand_landmarks):
    """ Determines if a detected hand is a closed fist. """
    if not hand_landmarks:
        return False
    
    # Check if fingertips are curled in past their PIP joints
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    pip_joints = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP
    ]
    
    fingers_curled = all(
        hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y 
        for tip, pip in zip(finger_tips, pip_joints)
    )

    # Check if thumb is tucked in
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    thumb_tucked = math.hypot(thumb_tip.x - index_mcp.x, thumb_tip.y - index_mcp.y) < 0.1

    return fingers_curled and thumb_tucked

# --- 4. Main Application Logic ---
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    # Variables for prediction logic
    sequence_buffer = []
    last_punch_time = 0
    punch_cooldown = 1.0  # 1 second cooldown
    last_hand_state = 'OPEN'
    current_prediction = "---"

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Make detections
        rgb_frame.flags.writeable = False
        pose_result = pose.process(rgb_frame)
        hands_result = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Extract keypoints
        keypoints = extract_keypoints(pose_result, hands_result)
        sequence_buffer.append(keypoints)
        sequence_buffer = sequence_buffer[-sequence_length:] # Keep buffer at fixed size

        # --- Prediction Logic ---
        hand_detected = hands_result.multi_hand_landmarks is not None
        current_hand_state = 'FIST' if hand_detected and is_fist(hands_result.multi_hand_landmarks[0]) else 'OPEN'
        current_time = time.time()

        # Trigger on the transition from OPEN to FIST and if cooldown is over
        if current_hand_state == 'FIST' and last_hand_state == 'OPEN' and (current_time - last_punch_time > punch_cooldown):
            if len(sequence_buffer) == sequence_length:
                # Prepare data for model
                input_data = np.expand_dims(sequence_buffer, axis=0)
                input_data = input_data.reshape(1, -1) # Flatten the sequence

                # Make prediction
                prediction = model.predict(input_data)
                confidence = model.predict_proba(input_data)
                
                predicted_action = actions[prediction[0]]
                pred_confidence = np.max(confidence)
                
                # Update display text only if confidence is high
                if pred_confidence > 0.7: # Confidence threshold
                    current_prediction = f"{predicted_action.upper()} ({pred_confidence:.2f})"
                    last_punch_time = current_time # Reset cooldown timer
                else:
                    current_prediction = "Uncertain..."

        last_hand_state = current_hand_state

        # --- Visualization ---
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if pose_result.pose_landmarks:
            mp_drawing.draw_landmarks(bgr_frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if hands_result.multi_hand_landmarks:
            for hand_landmarks in hands_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(bgr_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display Prediction
        cv2.rectangle(bgr_frame, (0, 0), (640, 60), (245, 117, 16), -1)
        cv2.putText(bgr_frame, 'LAST PUNCH', (15, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr_frame, current_prediction, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Real-Time Boxing Analysis", bgr_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()

if __name__ == '__main__':
    main()