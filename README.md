# 🥊 A Hybrid YOLO-LSTM Model with Real-Time Boxing Action Classification

This project is a comprehensive real-time boxing analysis system that uses computer vision and deep learning to classify boxing techniques, detect stances, and provide interactive feedback. It leverages a YOLO model for pose estimation and custom-trained LSTM models for action classification.

## ✨ Features

- **👊 Real-Time Punch Classification**: Identifies Jabs, Crosses, Hooks, and Uppercuts.
- **🛡️ Real-Time Block Classification**: Recognizes Parries, High Guards, and Forearm Blocks.
- **🧍 Stance Detection**: Automatically determines if a boxer is in an Orthodox or Southpaw stance.
- **🎯 Hit & Miss Logic**: A basic system to evaluate if a punch landed or was blocked/missed.
- **💻 Interactive UI**: Displays real-time scores, actions, and player states.
- **🔊 Audio Feedback**: Provides vocal encouragement and confirmation of detected punches.
- **🔗 Comprehensive Data Pipeline**: Includes scripts for data collection, augmentation, and processing.

## 🚀 Project Pipeline

The project follows a complete machine learning pipeline, from data collection to real-time deployment.

### 1. 📹 Data Collection
- **Source**: Video data was recorded using four IP webcams to capture different angles.
- **Punch Classes**: `jab`, `cross`, `hook`, `uppercut`, and a `negative` class for non-punching actions.
- **Block Classes**: `parry`, `high_guard`, `forearm_block`, and a `negative` class.
- **Details**: Each action was recorded for both `orthodox` and `southpaw` stances, using both `left` and `right` hands, resulting in a detailed and balanced raw dataset. Each clip is 3 seconds long at 20 FPS.

### 2. 🎨 Data Augmentation
To create a robust model, the raw video data was augmented using various techniques:
- **Rotation**: ±15 degrees.
- **Scaling**: 80% and 120% of the original size.
- **Noise**: Gaussian noise addition.
- **Brightness**: ±40 brightness adjustment.
- **Slow Motion**: 50% speed reduction.
- **Random Erase**: Occluding a random 15% patch of the frame.

### 3. ↔️ Data Mirroring
- All raw and augmented videos were horizontally flipped (`cv2.flip`).
- This technique effectively doubled the dataset size and ensured the model could generalize to both left-handed and right-handed stances regardless of the original recording.

### 4. 💪 Keypoint Extraction
- **Model**: A **YOLOv11m-pose** model was used to extract 17 body keypoints from each frame of every video.
- **Processing**:
    - For action classification, 8 keypoints from the upper body were selected (shoulders, elbows, wrists, hips).
    - The keypoints were normalized based on the distance between the shoulders to make the model scale-invariant.
    - Each video was converted into a sequence of 25 frames of normalized keypoints, padded with zeros if shorter.
- **Output**: The processed data for each video was saved as a `.npy` file.

### 5. 🧠 Model Training
- **Architecture**: Two separate **LSTM (Long Short-Term Memory)** networks were trained: one for classifying punches and one for blocks.
    - **Layers**: 2 LSTM layers with a dropout of 0.5.
    - **Hidden Size**: 128 units.
    - **Output**: A fully connected layer mapping to the number of classes.
- **📈 Performance**:
    - **Punch Classifier**: Achieved a validation accuracy of **99.74%**.
    - **Block Classifier**: Achieved a validation accuracy of **99.28%**.

## 🛠️ How to Run

### ✅ Prerequisites
- **Shell Environment**: The setup commands are intended for a Unix-like shell environment (e.g., Bash on Linux/macOS or WSL on Windows).
- **Python**: Python 3.10+
- **NVIDIA GPU**: (Optional, but highly recommended for performance) An NVIDIA GPU with CUDA support.

### ⚙️ Setup
1.  **Clone the Repository**:
    ```
    git clone https://github.com/ACM40960/Boxing.git
    cd Boxing
    ```

2.  **Create a Virtual Environment**:
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **CUDA Installation (Optional)**:
    For GPU acceleration, you need to install the NVIDIA CUDA Toolkit.
    - **Driver**: Ensure you have the latest NVIDIA drivers installed for your GPU.
    - **Toolkit**: Download and install the CUDA Toolkit that is compatible with your driver and the required version for PyTorch. You can find this on the NVIDIA CUDA Toolkit Archive.
    - **Verification**: You can verify the installation by running `nvcc --version` in your terminal.

4.  **Install Python Packages**:
    - **For GPU (with CUDA)**:
      First, visit the PyTorch official website to find the correct command for your specific CUDA version. For example, for CUDA 11.8, the command is:
      ```
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
      ```
      Then, install the remaining packages:
      ```
      pip install ultralytics opencv-python numpy pandas scikit-learn matplotlib seaborn pygame
      ```

    - **For CPU Only**:
      If you do not have a compatible GPU, install the CPU-only version of PyTorch and other packages:
      ```
      pip install torch torchvision torchaudio
      pip install ultralytics opencv-python numpy pandas scikit-learn matplotlib seaborn pygame
      ```

5.  **Download Models**:
    - Download the YOLO pose model: `yolo11m-pose.pt`
    - Place the trained LSTM models (`punch_classifier.pth`, `block_classifier.pth`) in the root directory.

6.  **Audio Files**:
    - Create an `audio` folder in the root directory.
    - Place `.wav` or `.mp3` files for the different punch types inside, following the names in the `AUDIO_FILE_MAPPING` dictionary in the final script.

### ▶️ Running the Application
The main application is the "Shadow Boxing Trainer". Execute the final cell in the `script.ipynb` notebook or run it as a Python script.

**Controls**:
- `q`: Quit the application.
- `s`: Toggle audio feedback ON/OFF.
- `+` / `-`: Increase/decrease audio volume.

## 📁 Project Structure

```bash
project-root/
├── script.ipynb
├── punch_classifier.pth
├── block_classifier.pth
├── yolo11m-pose.pt
├── audio/
│   ├── CLEAN JAB.wav
│   └── ... (other audio files)
├── punch/
│   ├── combined_dataset/
│   ├── augmented_data/
│   ├── mirrored_combined_dataset/
│   ├── mirrored_augmented_data/
│   └── processed_keypoints/
├── block_negative/
│   ├── block_dataset/
│   ├── block_augmented_data/
│   ├── mirrored_block_dataset/
│   ├── mirrored_block_augmented_data/
│   └── processed_keypoints/
```


## ⚠️ Disclaimer
This project is for educational and demonstrative purposes only. It is not a substitute for professional boxing instruction. Always consult with a qualified coach before engaging in any physical training.
