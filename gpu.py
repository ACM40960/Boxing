import tensorflow as tf
import subprocess
import re

# 1. Check for GPU availability using TensorFlow
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("Num GPUs Available: ", len(gpus))
    print("Detected GPU(s) by TensorFlow:")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name} (Type: {gpu.device_type})")

    # 2. Try to get the human-readable GPU name using nvidia-smi
    try:
        # Run nvidia-smi command to get GPU names
        # 'vc_redist.x86.exe' from the response history should resolve the error.
        smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], encoding='utf-8')
        gpu_names = [name.strip() for name in smi_output.strip().split('\n')]

        if gpu_names:
            print("\nHuman-readable GPU Name(s) from nvidia-smi:")
            for i, name in enumerate(gpu_names):
                print(f"  GPU {i}: {name}")
        else:
            print("\nCould not retrieve human-readable GPU name from nvidia-smi.")

    except FileNotFoundError:
        print("\nnvidia-smi not found. Please ensure NVIDIA drivers are installed and in PATH.")
    except Exception as e:
        print(f"\nAn error occurred while running nvidia-smi: {e}")
else:
    print("No GPUs Available.")