import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from collections import deque
from pathlib import Path
from enum import Enum, auto
import time
import threading
import pygame
from queue import Queue
import random
import os

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
    "RECOVERY_FRAMES": 60,
    "PREVIEW_WIDTH": 1280,
    "PREVIEW_HEIGHT": 720,
    "ACTION_TEXT_FADE_FRAMES": 90,
    "FPS_TARGET": 30,
    "MIN_PUNCH_SPEED": 2.0,
    "COMBO_TIMEOUT": 3.0,
    "AUDIO_ENABLED": True,
    "ENCOURAGEMENT_FREQUENCY": 5,
    "TECHNIQUE_FEEDBACK": True,
    "AUDIO_DIR": "app/static/audio",
    "AUDIO_VOLUME": 1.0,
    "AUDIO_FORMAT": [".wav", ".mp3", ".ogg"],
    "AUDIO_COOLDOWN": 2.0
}

AUDIO_FILE_MAPPING = {
    'jab': ['CLEAN JAB.wav','SHARP JAB.wav', 'GOOD JAB.wav','NICE JAB.wav'],
    'cross': ['POWERFUL CROSS.wav','GOOD CROSS.wav','NICE CROSS.wav','STRONG CROSS.wav'],
    'hook': ['SOLID HOOK.wav','GOOD HOOK.wav','NICE HOOK.wav'],
    'uppercut': ['Solid uppercut.wav','Good uppercut.wav','Nice uppercut.wav','Strong uppercut.wav'],
}

class BoxerState(Enum):
    IDLE = auto()
    PUNCHING = auto()
    RECOVERING = auto()

class AudioFeedback:
    def __init__(self):
        try:
            pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(8)
        except pygame.error as e:
            print(f"Pygame mixer could not be initialized: {e}. Audio will be disabled.")
            config["AUDIO_ENABLED"] = False
            return
        
        self.audio_queue = Queue()
        self.audio_files = {}
        self.audio_dir = Path(config["AUDIO_DIR"])
        self.last_played_time = 0
        self.punch_channel = pygame.mixer.Channel(0)

        self._load_audio_files()
        
        if config["AUDIO_ENABLED"]:
            self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
            self.audio_thread.start()
            total_files = sum(len(files) for files in self.audio_files.values())
            print(f"Audio system initialized. Loaded {total_files} audio files.")
            self._print_loaded_files()

    def _load_audio_files(self):
        if not self.audio_dir.is_dir():
            print(f"Audio directory '{self.audio_dir}' not found. Audio will be disabled.")
            print(f"Checking audio directory: {self.audio_dir.resolve()}")
            config["AUDIO_ENABLED"] = False
            return
        
        for category in AUDIO_FILE_MAPPING.keys():
            self.audio_files[category] = []
            
        for category, filenames in AUDIO_FILE_MAPPING.items():
            for filename in filenames:
                file_path = self.audio_dir / filename
                if file_path.exists():
                    try:
                        sound = pygame.mixer.Sound(str(file_path))
                        sound.set_volume(config["AUDIO_VOLUME"])
                        self.audio_files[category].append(sound)
                    except pygame.error as e:
                        print(f"Could not load '{file_path}': {e}")
                else:
                    print(f"File not found, skipping: {file_path}")
        
        self._create_fallbacks()

    def _create_fallbacks(self):
        if 'combo' not in self.audio_files or not self.audio_files.get('combo'):
            all_punch_sounds = [s for k in ['jab', 'cross', 'hook', 'uppercut'] for s in self.audio_files.get(k, [])]
            self.audio_files['combo'] = all_punch_sounds
        if 'encouragement' not in self.audio_files or not self.audio_files.get('encouragement'):
            all_sounds = [s for cat_sounds in self.audio_files.values() for s in cat_sounds]
            self.audio_files['encouragement'] = random.sample(all_sounds, min(len(all_sounds), 5)) if all_sounds else []
        if 'technique' not in self.audio_files or not self.audio_files.get('technique'):
            all_sounds = [s for cat_sounds in self.audio_files.values() for s in cat_sounds]
            self.audio_files['technique'] = random.sample(all_sounds, min(len(all_sounds), 5)) if all_sounds else []

    def _print_loaded_files(self):
        print("\nAudio Files Summary:")
        for category, sounds in self.audio_files.items():
            print(f"  - {category.upper()}: {len(sounds)} files")
        print()

    def _audio_worker(self):
        while True:
            category = self.audio_queue.get()
            if category in self.audio_files and self.audio_files[category]:
                try:
                    sound = random.choice(self.audio_files[category])
                    if category in config["PUNCH_ACTIONS"]:
                        self.punch_channel.play(sound)
                    else:
                        found_channel = pygame.mixer.find_channel(True)
                        if found_channel:
                           found_channel.play(sound)
                except Exception as e:
                    print(f"Error playing audio: {e}")
            self.audio_queue.task_done()

    def play_audio(self, category):
        current_time = time.time()
        if (config["AUDIO_ENABLED"] and 
            category in self.audio_files and
            current_time - self.last_played_time > config["AUDIO_COOLDOWN"]):
            
            if self.audio_files[category]:
                while not self.audio_queue.empty():
                    try: self.audio_queue.get_nowait()
                    except queue.Empty: break
                self.audio_queue.put(category)
                self.last_played_time = current_time

    def set_volume(self, volume):
        new_volume = max(0.0, min(1.0, volume))
        config["AUDIO_VOLUME"] = new_volume
        for category_sounds in self.audio_files.values():
            for sound in category_sounds:
                sound.set_volume(new_volume)


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
    if len(buffer) < 5: return 0.0
    k_map = config["KEYPOINT_MAP"]
    recent_frames = list(buffer)[-5:]
    speeds = []
    for wrist_key in ['left_wrist', 'right_wrist']:
        wrist_idx = k_map[wrist_key]
        positions = [f["raw_kps"][wrist_idx] for f in recent_frames if f["confs"][wrist_idx] > config["KEYPOINT_CONF_THRESH"]]
        if len(positions) >= 2:
            total_distance = sum(np.linalg.norm(positions[i+1] - positions[i]) for i in range(len(positions)-1))
            speeds.append(total_distance / len(positions))
    return max(speeds) if speeds else 0.0

class ShadowBoxingAnalyzer:
    def __init__(self, models, audio_feedback):
        self.models = models
        self.audio = audio_feedback
        self.boxer_data = {
            "buffer": deque(maxlen=config["SEQUENCE_LENGTH"]),
            "state": BoxerState.IDLE,
            "recovery_timer": 0,
            "action_text": "",
            "action_timer": 0,
            "punch_count": 0,
            "combo_count": 0,
            "last_punch_time": 0,
            "combo_history": deque(maxlen=5),
            "stance": "Unknown",
            "session_start": time.time()
        }
    
    def update_boxer_data(self, kps, confs):
        norm_kps = normalize_keypoints(kps, confs)
        if norm_kps is not None:
            self.boxer_data["buffer"].append({
                "norm_kps": norm_kps,
                "raw_kps": kps.copy(),
                "confs": confs.copy(),
                "timestamp": time.time()
            })
        self.boxer_data['stance'] = detect_stance(kps, confs)
    
    def analyze_action(self, device):
        if (self.boxer_data["state"] == BoxerState.IDLE and 
            len(self.boxer_data["buffer"]) == config["SEQUENCE_LENGTH"]):
            
            if len(self.boxer_data["buffer"][0]["norm_kps"]) != 16: return
            
            sequence = np.array([f["norm_kps"] for f in self.boxer_data["buffer"]])
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                punch_output = self.models['punch'](sequence_tensor)
                punch_conf, punch_idx = torch.max(torch.softmax(punch_output, dim=1), 1)
                
                if punch_conf.item() > config["CONFIDENCE_THRESHOLD"]:
                    action_name = config["PUNCH_ACTIONS"][punch_idx.item()]
                    if calculate_punch_speed(self.boxer_data["buffer"]) > config["MIN_PUNCH_SPEED"]:
                        self._handle_punch_detected(action_name, punch_conf.item())
    
    def _handle_punch_detected(self, punch_type, confidence):
        current_time = time.time()
        self.boxer_data["punch_count"] += 1
        is_combo = (current_time - self.boxer_data["last_punch_time"]) < config["COMBO_TIMEOUT"]
        self.boxer_data["combo_count"] = self.boxer_data["combo_count"] + 1 if is_combo else 1
        self.boxer_data["last_punch_time"] = current_time
        self.boxer_data["combo_history"].append(punch_type)
        
        quality_text = "EXCELLENT" if confidence > 0.9 else "GOOD" if confidence > 0.8 else "NICE"
        self.boxer_data["action_text"] = f'{quality_text} {punch_type.upper()}!'
        self.boxer_data["action_timer"] = config["ACTION_TEXT_FADE_FRAMES"]
        self.boxer_data["state"] = BoxerState.RECOVERING
        self.boxer_data["recovery_timer"] = config["RECOVERY_FRAMES"]
        self._provide_audio_feedback(punch_type)
    
    def _provide_audio_feedback(self, punch_type):
        if self.boxer_data["combo_count"] >= 3: self.audio.play_audio('combo')
        elif self.boxer_data["punch_count"] % config["ENCOURAGEMENT_FREQUENCY"] == 0: self.audio.play_audio('encouragement')
        else: self.audio.play_audio(punch_type)
    
    def update_state(self):
        if self.boxer_data["state"] == BoxerState.RECOVERING:
            self.boxer_data["recovery_timer"] -= 1
            if self.boxer_data["recovery_timer"] <= 0:
                self.boxer_data["state"] = BoxerState.IDLE
        if self.boxer_data["action_timer"] > 0:
            self.boxer_data["action_timer"] -= 1

def draw_shadow_boxing_ui(frame, analyzer, detection=None):
    h, w, _ = frame.shape
    boxer_data = analyzer.boxer_data
    
    hud_h = 170
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - hud_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    if detection:
        box = detection["box"].astype(int)
        kps = detection["kps"]
        confs = detection["confs"]
        
        color = (0, 255, 0)
        if boxer_data["state"] == BoxerState.PUNCHING: color = (0, 0, 255)
        elif boxer_data["state"] == BoxerState.RECOVERING: color = (0, 255, 255)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        k_map = config["KEYPOINT_MAP"]
        for name in ['left_wrist', 'right_wrist', 'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder']:
            idx = k_map[name]
            if confs[idx] > config["KEYPOINT_CONF_THRESH"]:
                point = (int(kps[idx][0]), int(kps[idx][1]))
                cv2.circle(frame, point, 8, (0, 255, 255), -1)
        
        if boxer_data["action_timer"] > 0:
            cv2.putText(frame, boxer_data["action_text"], (box[0], box[1] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
    
    session_time = int(time.time() - boxer_data["session_start"])
    minutes, seconds = divmod(session_time, 60)
    stats_y = h - 140
    cv2.putText(frame, f"Session Time: {minutes:02d}:{seconds:02d}", (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Total Punches: {boxer_data['punch_count']}", (20, stats_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Current Combo: {boxer_data['combo_count']}", (20, stats_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Stance: {boxer_data['stance']}", (20, stats_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    audio_status = "ON" if config["AUDIO_ENABLED"] else "OFF"
    audio_color = (0, 255, 0) if config["AUDIO_ENABLED"] else (0, 0, 255)
    cv2.putText(frame, f"Audio: {audio_status} | Vol: {int(config['AUDIO_VOLUME']*100)}%", (20, stats_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, audio_color, 2)
    
    state_color = (0, 255, 0) if boxer_data["state"] == BoxerState.IDLE else (0, 0, 255) if boxer_data["state"] == BoxerState.PUNCHING else (0, 255, 255)
    cv2.putText(frame, f"State: {boxer_data['state'].name}", (w - 300, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
    
    if boxer_data["combo_history"]:
        combo_text = " → ".join(list(boxer_data["combo_history"])[-3:])
        cv2.putText(frame, f"Recent: {combo_text}", (w - 400, stats_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    return frame


print("Loading shadow boxing models and audio...")
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
audio_feedback = AudioFeedback()
analyzer = ShadowBoxingAnalyzer(models, audio_feedback)
print("Shadow boxing models and audio loaded successfully.")


def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    analyzer.boxer_data["session_start"] = time.time()
    analyzer.boxer_data["punch_count"] = 0
    analyzer.boxer_data["combo_history"].clear()
    analyzer.boxer_data["combo_count"] = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        yolo_results = yolo_model(frame, verbose=False, device=device)
        
        current_detection = None
        if yolo_results and yolo_results[0].boxes is not None and len(yolo_results[0].boxes) > 0:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            keypoints_list = yolo_results[0].keypoints.data.cpu().numpy()
            if len(boxes) > 0 and len(keypoints_list) > 0:
                current_detection = {
                    "kps": keypoints_list[0][:, :2],
                    "confs": keypoints_list[0][:, 2],
                    "box": boxes[0]
                }
                analyzer.update_boxer_data(current_detection["kps"], current_detection["confs"])

        analyzer.analyze_action(device)
        analyzer.update_state()
        
        display_frame = draw_shadow_boxing_ui(frame, analyzer, current_detection)

        (flag, encodedImage) = cv2.imencode(".jpg", display_frame)
        if not flag:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

    cap.release()
