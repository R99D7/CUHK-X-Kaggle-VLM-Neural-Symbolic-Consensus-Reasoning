import os
import torch
import torchvision
import numpy as np
from torchvision.models.video import r3d_18, R3D_18_Weights
import cv2
from tqdm import tqdm
import pandas as pd

# Configurations
NUM_FRAMES = 16
RESIZE_DIM = 224
BATCH_SIZE = 4  # Very small batch size for 6GB VRAM just for extraction
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = 'video_features_r3d'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Pretrained 3D-CNN Model
print(f"Loading r3d_18 model on {DEVICE}...")
weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
# Remove the final fully connected layer to get the 512-dim feature vector
model.fc = torch.nn.Identity()
model = model.to(DEVICE)
model.eval()

# Normalization stats for Kinetics-400
mean = [0.43216, 0.394666, 0.37645]
std = [0.22803, 0.22145, 0.216989]

def get_sparse_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return None
        
    # Uniformly sample frame indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # OpenCV loads as BGR, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (RESIZE_DIM, RESIZE_DIM))
            frames.append(frame)
        else:
            # Fallback if frame read fails (duplicate last valid frame)
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((RESIZE_DIM, RESIZE_DIM, 3), dtype=np.uint8))
                
    cap.release()
    
    # Shape: (T, H, W, C) -> (C, T, H, W) for PyTorch 3D CNNs
    frames = np.array(frames)
    frames = frames.transpose(3, 0, 1, 2)
    
    # Normalize to [0, 1]
    tensor = torch.FloatTensor(frames) / 255.0
    
    # Apply standard normalization per channel
    for i in range(3):
        tensor[i] = (tensor[i] - mean[i]) / std[i]
        
    return tensor

def process_video_folder(base_dir, split_name):
    print(f"Processing {split_name} videos in {base_dir}...")
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found. Skipping.")
        return
        
    video_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    for folder in tqdm(video_folders, desc=f"Extracting {split_name}"):
        vid_id = folder
        out_path = os.path.join(OUTPUT_DIR, f"{vid_id}.pt")
        
        if os.path.exists(out_path):
            continue
            
        # Target the Depth.mp4 video as requested by blueprint (pseudo-RGB)
        vid_path = os.path.join(base_dir, folder, "Depth", "Depth.mp4")
        if not os.path.exists(vid_path):
            # Fallback to Depth_Color if raw Depth is missing
            vid_path = os.path.join(base_dir, folder, "Depth_Color", "Depth_Color.mp4")
            
        if not os.path.exists(vid_path):
            continue
            
        # 1. Load and sample frames
        video_tensor = get_sparse_frames(vid_path, NUM_FRAMES)
        if video_tensor is None:
            continue
            
        # 2. Add batch dimension: (1, C, T, H, W)
        video_tensor = video_tensor.unsqueeze(0).to(DEVICE)
        
        # 3. Forward pass
        with torch.no_grad():
            features = model(video_tensor) # Shape: [1, 512]
            
        # 4. Save feature vector
        features = features.cpu().squeeze(0) # Shape: [512]
        torch.save(features, out_path)

if __name__ == '__main__':
    # We will process both Train and Test videos
    
    # 1. Test Videos
    test_dir = os.path.join('videos', 'large_model_track_test')
    process_video_folder(test_dir, 'Test')
    
    # 2. Train Videos (Assuming HARn and HAU get extracted directly to their folder names)
    # The actual folder structure of HARn.zip / HAU.zip is likely to just have the video folders.
    train_dir_harn = os.path.join('videos', 'HARn')
    train_dir_hau = os.path.join('videos', 'HAU')
    
    process_video_folder(train_dir_harn, 'Train_HARn')
    process_video_folder(train_dir_hau, 'Train_HAU')
    
    print("Feature extraction complete!")
