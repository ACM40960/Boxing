import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from collections import deque
from pathlib import Path
from enum import Enum, auto
import time

config = {
    "ROOT_DIR": Path.cwd(),
    "YOLO_MODEL_NAME": "yolo11m-pose.pt",
    "PUNCH_LSTM_MODEL_NAME": "punch_classifier.pth",
    "BLOCK_LSTM_MODEL_NAME": "block_classifier.pth",
    "PUNCH_ACTIONS": ['jab', 'cross', 'hook', 'uppercut'],
    "BLOCK_ACTIONS": ['forearm_block', 'high_guard', 'parry', 'negative'],
    "KEYPOINT_MAP": {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
    },
    "ACTION_KEYPOINTS": [
        'left_shoulder', 'right_shoulder', 
        'left_elbow', 'right_elbow', 
        'left_wrist', 'right_wrist',
        'left_hip', 'right_hip'
    ],
    "SEQUENCE_LENGTH": 25,
    "CONFIDENCE_THRESHOLD": 0.7,
    "KEYPOINT_CONF_THRESH": 0.5,
    "PLAYER_TIMEOUT_FRAMES": 300,  # Increased for better tracking stability
    "RECOVERY_FRAMES": 60,
    "PREVIEW_WIDTH": 1280,
    "PREVIEW_HEIGHT": 720,
    "ACTION_TEXT_FADE_FRAMES": 90,
    "MIN_PUNCH_SPEED": 2.0,        # For initial motion detection
    "MIN_PUNCH_VELOCITY": 1.0,     # For peak frame velocity check
    "COMBO_TIMEOUT": 1.5,          # Configurable combo window
    "FPS_TARGET": 30,
    "HIT_DISTANCE_THRESHOLD": 30   # Threshold for expanded hit detection (pixels)
}

class PlayerState(Enum):
    IDLE = auto()
    ATTACKING = auto()
    BLOCKING = auto()
    RECOVERING = auto()

class ActionClassifierLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(ActionClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def normalize_keypoints(kps, confs):
    k_map = config["KEYPOINT_MAP"]
    ls_idx, rs_idx = k_map['left_shoulder'], k_map['right_shoulder']
    
    if confs[ls_idx] < config["KEYPOINT_CONF_THRESH"] or confs[rs_idx] < config["KEYPOINT_CONF_THRESH"]:
        return None
    
    left_shoulder, right_shoulder = kps[ls_idx], kps[rs_idx]
    scale_dist = np.linalg.norm(left_shoulder - right_shoulder)
    if scale_dist < 1e-4: 
        return None
    
    selected_indices = [k_map[name] for name in config["ACTION_KEYPOINTS"]]
    kps_subset = kps[selected_indices]
    confs_subset = confs[selected_indices]
    
    center_point = (left_shoulder + right_shoulder) / 2
    normalized_kps = np.zeros_like(kps_subset)
    valid_mask = confs_subset > config["KEYPOINT_CONF_THRESH"]
    normalized_kps[valid_mask] = (kps_subset[valid_mask] - center_point) / scale_dist
    
    return normalized_kps.flatten()

def detect_stance(kps, confs):
    k_map = config["KEYPOINT_MAP"]
    required_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    if any(confs[k_map[name]] < config["KEYPOINT_CONF_THRESH"] for name in required_points):
        return "Unknown"
        
    left_shoulder, right_shoulder = kps[k_map['left_shoulder']], kps[k_map['right_shoulder']]
    left_hip, right_hip = kps[k_map['left_hip']], kps[k_map['right_hip']]
    dist_left_side = np.linalg.norm(left_shoulder - left_hip)
    dist_right_side = np.linalg.norm(right_shoulder - right_hip)
    return 'Southpaw' if dist_left_side < dist_right_side else 'Orthodox'

def calculate_punch_speed(buffer):
    """Calculate punch speed based on wrist movement"""
    if len(buffer) < 5: 
        return 0.0
    
    k_map = config["KEYPOINT_MAP"]
    recent_frames = list(buffer)[-5:]
    speeds = []
    
    for wrist_key in ['left_wrist', 'right_wrist']:
        wrist_idx = k_map[wrist_key]
        positions = [f["raw_kps"][wrist_idx] for f in recent_frames 
                     if f["confs"][wrist_idx] > config["KEYPOINT_CONF_THRESH"]]
        
        if len(positions) >= 2:
            total_distance = sum(np.linalg.norm(positions[i+1] - positions[i]) 
                                 for i in range(len(positions)-1))
            speeds.append(total_distance / len(positions))
    
    return max(speeds) if speeds else 0.0

def find_peak_extension_frame(buffer, wrist_idx, shoulder_idx):
    max_extension = 0
    peak_frame = -1
    
    for i, frame in enumerate(buffer):
        wrist_pos = frame["raw_kps"][wrist_idx]
        shoulder_pos = frame["raw_kps"][shoulder_idx]
        
        if np.all(wrist_pos == 0) or np.all(shoulder_pos == 0):
            continue
            
        extension = np.linalg.norm(wrist_pos - shoulder_pos)
        if extension > max_extension:
            max_extension = extension
            peak_frame = i
    
    return peak_frame if peak_frame != -1 else len(buffer) - 1

def is_hit(attacker_buffer, defender_buffer, attacker_id, defender_id):
    if len(attacker_buffer) < 5 or len(defender_buffer) < 1:
        return False

    k_map = config["KEYPOINT_MAP"]
    
    # 1. Determine dominant punching hand based on movement
    start_frame = attacker_buffer[0]
    end_frame = attacker_buffer[-1]
    
    left_wrist_delta = np.linalg.norm(
        end_frame["raw_kps"][k_map['left_wrist']] - start_frame["raw_kps"][k_map['left_wrist']]
    ) if (start_frame["confs"][k_map['left_wrist']] > config["KEYPOINT_CONF_THRESH"] and
           end_frame["confs"][k_map['left_wrist']] > config["KEYPOINT_CONF_THRESH"]) else 0
           
    right_wrist_delta = np.linalg.norm(
        end_frame["raw_kps"][k_map['right_wrist']] - start_frame["raw_kps"][k_map['right_wrist']]
    ) if (start_frame["confs"][k_map['right_wrist']] > config["KEYPOINT_CONF_THRESH"] and
           end_frame["confs"][k_map['right_wrist']] > config["KEYPOINT_CONF_THRESH"]) else 0
    
    # Determine which hand is punching
    if left_wrist_delta > right_wrist_delta:
        wrist_idx = k_map['left_wrist']
        shoulder_idx = k_map['left_shoulder']
    else:
        wrist_idx = k_map['right_wrist']
        shoulder_idx = k_map['right_shoulder']
    
    # 2. Find peak extension frame for the punching hand
    peak_idx = find_peak_extension_frame(attacker_buffer, wrist_idx, shoulder_idx)
    
    # 3. Velocity check at peak frame
    if peak_idx >= 2:
        velocity = np.linalg.norm(
            attacker_buffer[peak_idx]["raw_kps"][wrist_idx] - 
            attacker_buffer[peak_idx-2]["raw_kps"][wrist_idx]
        ) / 2
        if velocity < config["MIN_PUNCH_VELOCITY"]:
            return False
    
    attacker_kps = attacker_buffer[peak_idx]["raw_kps"]
    attacker_confs = attacker_buffer[peak_idx]["confs"]
    defender_kps = defender_buffer[-1]["raw_kps"]
    defender_confs = defender_buffer[-1]["confs"]
    
    target_zones = {
        "head": ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'],
        "torso": ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    }
    
    for zone_name, keypoint_names in target_zones.items():
        zone_points = []
        for name in keypoint_names:
            idx = k_map.get(name)
            if idx is None:
                continue
            if defender_confs[idx] > config["KEYPOINT_CONF_THRESH"]:
                zone_points.append(defender_kps[idx])
        
        if len(zone_points) < 3:
            continue
        
        hull = cv2.convexHull(np.array(zone_points, dtype=np.float32))
        
        if attacker_confs[wrist_idx] > config["KEYPOINT_CONF_THRESH"]:
            wrist_point = tuple(attacker_kps[wrist_idx].astype(np.float32))
            # Enhanced: Use measure=True for distance; hit if inside or within threshold outside
            dist = cv2.pointPolygonTest(hull, wrist_point, measureDist=True)
            if dist >= -config["HIT_DISTANCE_THRESHOLD"]:
                return True
                
    return False

def update_player_data(detections, player_data, player_roles, last_seen, frame_counter):
    for track_id, data in detections.items():
        if track_id not in player_data:
            if len(player_roles) >= 2:
                continue
            available_roles = {1, 2} - set(player_roles.values())
            if not available_roles:
                continue
            role_id = min(available_roles)
            player_roles[track_id] = role_id
            player_data[track_id] = {
                "buffer": deque(maxlen=config["SEQUENCE_LENGTH"]),
                "state": PlayerState.IDLE,
                "recovery_timer": 0,
                "action_text": "",
                "last_action": "",
                "action_timer": 0,
                "hits": 0,
                "misses": 0,
                "stance": "Unknown",
                "combo_count": 0,
                "last_action_time": 0
            }
        
        kps = data["kps"]
        confs = data["confs"]
        
        norm_kps = normalize_keypoints(kps, confs)
        if norm_kps is not None:
            player_data[track_id]["buffer"].append({
                "norm_kps": norm_kps,
                "raw_kps": kps.copy(),
                "confs": confs.copy(),
                "timestamp": time.time()
            })
        
        last_seen[track_id] = frame_counter

def manage_player_timeouts(player_data, player_roles, last_seen, frame_counter):
    timed_out_ids = [tid for tid, frame in last_seen.items() 
                     if frame_counter - frame > config["PLAYER_TIMEOUT_FRAMES"]]
    for track_id in timed_out_ids:
        if track_id in player_data: 
            del player_data[track_id]
        if track_id in last_seen: 
            del last_seen[track_id]
        if track_id in player_roles: 
            del player_roles[track_id]

def evaluate_and_update_states(player_data, player_roles, models, device):
    tracked_ids = list(player_roles.keys())
    if len(tracked_ids) < 2:
        return

    current_time = time.time()  # Consistent timestamp for all players
    
    # Reset combos if timeout exceeded
    for player_id in tracked_ids:
        p_data = player_data[player_id]
        if current_time - p_data["last_action_time"] > config["COMBO_TIMEOUT"]:
            p_data["combo_count"] = 0

    potential_actions = {}
    for player_id in tracked_ids:
        p_data = player_data[player_id]
        
        # Only evaluate if player is idle and has a full buffer
        if (p_data["state"] == PlayerState.IDLE and 
            len(p_data["buffer"]) == config["SEQUENCE_LENGTH"]):
            
            # Skip if normalization failed
            if len(p_data["buffer"][0]["norm_kps"]) != 16:
                continue
            
            # Check for significant motion before running models
            punch_speed = calculate_punch_speed(p_data["buffer"])
            if punch_speed < config["MIN_PUNCH_SPEED"]:
                continue
                
            sequence = np.array([f["norm_kps"] for f in p_data["buffer"]])
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                punch_output = models['punch'](sequence_tensor)
                punch_conf, punch_idx = torch.max(torch.softmax(punch_output, dim=1), 1)
                
                if punch_conf.item() > config["CONFIDENCE_THRESHOLD"]:
                    action_name = config["PUNCH_ACTIONS"][punch_idx.item()]
                    potential_actions[player_id] = {
                        "type": "punch", 
                        "name": action_name,
                        "confidence": punch_conf.item()
                    }
                    player_data[player_id]["last_action"] = action_name
                else:
                    block_output = models['block'](sequence_tensor)
                    block_conf, block_idx = torch.max(torch.softmax(block_output, dim=1), 1)
                    
                    if block_conf.item() > config["CONFIDENCE_THRESHOLD"]:
                        action_name = config["BLOCK_ACTIONS"][block_idx.item()]
                        potential_actions[player_id] = {
                            "type": "block", 
                            "name": action_name,
                            "confidence": block_conf.item()
                        }
                        player_data[player_id]["last_action"] = action_name

    # Process detected actions
    p1_id, p2_id = tracked_ids[0], tracked_ids[1]
    p1_action = potential_actions.get(p1_id)
    p2_action = potential_actions.get(p2_id)
    
    if p1_action and p1_action["type"] == "punch":
        outcome = ""
        if p2_action and p2_action["type"] == "block":
            outcome = "BLOCKED"
        else:
            if is_hit(list(player_data[p1_id]["buffer"]), list(player_data[p2_id]["buffer"]), p1_id, p2_id):
                outcome = "HIT"
                player_data[p1_id]["hits"] += 1
            else:
                outcome = "MISS"
                player_data[p1_id]["misses"] += 1
        
        # Update combo tracking
        is_combo = current_time - player_data[p1_id]["last_action_time"] < config["COMBO_TIMEOUT"]
        player_data[p1_id]["combo_count"] = player_data[p1_id]["combo_count"] + 1 if is_combo else 1
        player_data[p1_id]["last_action_time"] = current_time
        
        # Show quality assessment
        quality = "EXCELLENT" if p1_action["confidence"] > 0.9 else "GOOD" if p1_action["confidence"] > 0.8 else "NICE"
        player_data[p1_id]["action_text"] = f'{quality} {p1_action["name"].upper()} ({outcome})'
        player_data[p1_id]["action_timer"] = config["ACTION_TEXT_FADE_FRAMES"]
        player_data[p1_id]["state"] = PlayerState.RECOVERING
        player_data[p1_id]["recovery_timer"] = config["RECOVERY_FRAMES"]
    
    if p2_action and p2_action["type"] == "punch":
        outcome = ""
        if p1_action and p1_action["type"] == "block":
            outcome = "BLOCKED"
        else:
            if is_hit(list(player_data[p2_id]["buffer"]), list(player_data[p1_id]["buffer"]), p2_id, p1_id):
                outcome = "HIT"
                player_data[p2_id]["hits"] += 1
            else:
                outcome = "MISS"
                player_data[p2_id]["misses"] += 1
        
        # Update combo tracking
        is_combo = current_time - player_data[p2_id]["last_action_time"] < config["COMBO_TIMEOUT"]
        player_data[p2_id]["combo_count"] = player_data[p2_id]["combo_count"] + 1 if is_combo else 1
        player_data[p2_id]["last_action_time"] = current_time
        
        # Show quality assessment
        quality = "EXCELLENT" if p2_action["confidence"] > 0.9 else "GOOD" if p2_action["confidence"] > 0.8 else "NICE"
        player_data[p2_id]["action_text"] = f'{quality} {p2_action["name"].upper()} ({outcome})'
        player_data[p2_id]["action_timer"] = config["ACTION_TEXT_FADE_FRAMES"]
        player_data[p2_id]["state"] = PlayerState.RECOVERING
        player_data[p2_id]["recovery_timer"] = config["RECOVERY_FRAMES"]
    
    for player_id, action in potential_actions.items():
        if action["type"] == "block":
            # Show quality assessment for blocks
            quality = "EXCELLENT" if action["confidence"] > 0.9 else "GOOD" if action["confidence"] > 0.8 else "NICE"
            player_data[player_id]["action_text"] = f'{quality} {action["name"].upper()}'
            player_data[player_id]["action_timer"] = config["ACTION_TEXT_FADE_FRAMES"]
            player_data[player_id]["state"] = PlayerState.BLOCKING
            player_data[player_id]["last_action_time"] = current_time

def draw_ui(frame, player_data, player_roles, detections):
    h, w, _ = frame.shape
    hud_h = 150
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - hud_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for track_id, role_id in player_roles.items():
        if track_id not in player_data or track_id not in detections:
            continue
            
        p_data = player_data[track_id]
        box = detections[track_id]["box"].astype(int)
        kps = detections[track_id]["kps"]
        confs = detections[track_id]["confs"]
        
        color = (0, 255, 0)
        if p_data["state"] == PlayerState.ATTACKING:
            color = (0, 0, 255)
        elif p_data["state"] == PlayerState.BLOCKING:
            color = (255, 0, 0)
        elif p_data["state"] == PlayerState.RECOVERING:
            color = (0, 255, 255)

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, f"Player {role_id}", (box[0], box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)    
        
        # Always show last action
        last_action = p_data.get("last_action", "").upper()
        cv2.putText(frame, f"Last: {last_action}", (box[0], box[1] - 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2, cv2.LINE_AA)
        
        # Show current action text if active
        if p_data["action_timer"] > 0:
            text_color = (100, 255, 100) if "HIT" in p_data["action_text"] else \
                         (100, 100, 255) if "MISS" in p_data["action_text"] else \
                         (255, 200, 100)
            cv2.putText(frame, p_data["action_text"], (box[0], box[1] - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3, cv2.LINE_AA)
        
        k_map = config["KEYPOINT_MAP"]
        target_zones = {
            "head": (['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'], (0, 255, 255)),
            "torso": (['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'], (255, 0, 255))
        }
        
        for zone_name, (keypoint_names, zone_color) in target_zones.items():
            zone_points = []
            for name in keypoint_names:
                idx = k_map.get(name)
                if idx is None: continue
                if confs[idx] > config["KEYPOINT_CONF_THRESH"]:
                    point = (int(kps[idx][0]), int(kps[idx][1]))
                    zone_points.append(point)
                    cv2.circle(frame, point, 5, zone_color, -1)
            
            if len(zone_points) >= 3:
                hull = cv2.convexHull(np.array(zone_points, dtype=np.int32))
                cv2.drawContours(frame, [hull], 0, zone_color, 2)
                
                if hull.size > 0:
                    M = cv2.moments(hull)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(frame, zone_name, (cX, cY), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_color, 2)

        x_pos = 50 if role_id == 1 else w - 450
        stance_text = f"Player {role_id} ({p_data.get('stance', 'N/A')})"
        cv2.putText(frame, stance_text, (x_pos, h - 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        score_text = f"Score: {p_data['hits']} / {p_data['misses']}"
        cv2.putText(frame, score_text, (x_pos, h - 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        combo_text = f"Combo: {p_data['combo_count']}"
        cv2.putText(frame, combo_text, (x_pos, h - 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        state_text = f"State: {p_data['state'].name}"
        cv2.putText(frame, state_text, (x_pos, h - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

    return frame

print("Loading two-player models...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
root = config["ROOT_DIR"]
yolo_model = YOLO(root / config["YOLO_MODEL_NAME"])

input_size = len(config["ACTION_KEYPOINTS"]) * 2
punch_model = ActionClassifierLSTM(input_size, 128, 2, len(config["PUNCH_ACTIONS"])).to(device)
punch_model.load_state_dict(torch.load(root / config["PUNCH_LSTM_MODEL_NAME"], map_location=device))
punch_model.eval()

block_model = ActionClassifierLSTM(input_size, 128, 2, len(config["BLOCK_ACTIONS"])).to(device)
block_model.load_state_dict(torch.load(root / config["BLOCK_LSTM_MODEL_NAME"], map_location=device))
block_model.eval()

models = {'punch': punch_model, 'block': block_model}
print("Two-player models loaded successfully.")

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    player_data = {}
    player_roles = {}
    last_seen = {}
    frame_counter = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        yolo_results = yolo_model.track(
            frame,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
            device=device
        )

        current_detections = {}
        if yolo_results and yolo_results[0].boxes and yolo_results[0].boxes.id is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            keypoints_list = yolo_results[0].keypoints.data.cpu().numpy()
            track_ids = yolo_results[0].boxes.id.int().cpu().tolist()

            for i, track_id in enumerate(track_ids):
                current_detections[track_id] = {
                    "kps": keypoints_list[i][:, :2],
                    "confs": keypoints_list[i][:, 2],
                    "box": boxes[i]
                }

        update_player_data(current_detections, player_data, player_roles, last_seen, frame_counter)
        manage_player_timeouts(player_data, player_roles, last_seen, frame_counter)

        for p_data in player_data.values():
            if p_data["state"] == PlayerState.RECOVERING:
                p_data["recovery_timer"] -= 1
                if p_data["recovery_timer"] <= 0:
                    p_data["state"] = PlayerState.IDLE
            elif p_data["state"] == PlayerState.BLOCKING:
                if time.time() - p_data.get("last_action_time", 0) > 0.5:
                    p_data["state"] = PlayerState.IDLE
        
        evaluate_and_update_states(player_data, player_roles, models, device)

        for track_id, p_data in player_data.items():
            if track_id in current_detections:
                p_data['stance'] = detect_stance(
                    current_detections[track_id]['kps'],
                    current_detections[track_id]['confs']
                )
            if p_data["action_timer"] > 0:
                p_data["action_timer"] -= 1

        display_frame = draw_ui(frame, player_data, player_roles, current_detections)

        (flag, encodedImage) = cv2.imencode(".jpg", display_frame)
        if not flag:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

    cap.release()
