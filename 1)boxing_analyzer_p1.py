import cv2
import mediapipe as mp
import numpy as np
import math
import time

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def is_fist(hand_landmarks):
    if not hand_landmarks:
        return False

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    finger_curled = (
        index_tip.y > index_pip.y and
        middle_tip.y > middle_pip.y and
        ring_tip.y > ring_pip.y and
        pinky_tip.y > pinky_pip.y
    )
    thumb_to_index_dist = math.hypot(thumb_tip.x - index_mcp.x, thumb_tip.y - index_mcp.y)
    thumb_tucked = thumb_to_index_dist < 0.1

    return finger_curled and thumb_tucked

def draw_torso_landmarks(frame, pose_landmarks):
    h, w, _ = frame.shape
    if not pose_landmarks:
        return

    # Extract important torso landmarks
    keypoints = {
        "RIGHT_SHOULDER": mp_pose.PoseLandmark.LEFT_SHOULDER,
        "LEFT_SHOULDER": mp_pose.PoseLandmark.RIGHT_SHOULDER,
        "RIGHT_HIP": mp_pose.PoseLandmark.LEFT_HIP,
        "LEFT_HIP": mp_pose.PoseLandmark.RIGHT_HIP
    }

    for label, landmark_enum in keypoints.items():
        landmark = pose_landmarks.landmark[landmark_enum]
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 6, (255, 255, 0), -1)
        cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    prev_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        hands_result = hands.process(rgb_frame)
        pose_result = pose.process(rgb_frame)

        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Draw Pose (torso) landmarks
        if pose_result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            draw_torso_landmarks(frame, pose_result.pose_landmarks)

        # Draw hand landmarks and detect fists
        if hands_result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hands_result.multi_hand_landmarks, hands_result.multi_handedness):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                hand_label = handedness.classification[0].label
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                h, w, _ = frame.shape
                coord_x = int(wrist.x * w)
                coord_y = int(wrist.y * h)

                cv2.putText(frame, hand_label, (coord_x - 50, coord_y + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if is_fist(hand_landmarks):
                    fist_status = "FIST"
                    color = (0, 255, 0)
                else:
                    fist_status = "OPEN"
                    color = (0, 0, 255)

                cv2.putText(frame, fist_status, (coord_x - 50, coord_y + 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Boxing Analysis - Hands + Torso", frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()

if __name__ == '__main__':
    main()