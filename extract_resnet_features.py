import os
import torch
import torchvision
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
import cv2
import pandas as pd
from tqdm import tqdm

# Configurations
NUM_FRAMES = 16
RESIZE_DIM = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = 'video_features_resnet'
MODALITIES_TO_EXTRACT = ['Depth', 'IR', 'Thermal']

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading ResNet50 model on {DEVICE}...")
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.fc = torch.nn.Identity()
model = model.to(DEVICE)
model.eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def get_sparse_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return None
        
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (RESIZE_DIM, RESIZE_DIM))
            frames.append(frame)
        else:
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((RESIZE_DIM, RESIZE_DIM, 3), dtype=np.uint8))
                
    cap.release()
    
    frames = np.array(frames)
    frames = frames.transpose(0, 3, 1, 2)
    
    tensor = torch.FloatTensor(frames) / 255.0
    for i in range(3):
        tensor[:, i, :, :] = (tensor[:, i, :, :] - mean[i]) / std[i]
        
    return tensor

def process_dataframe(csv_path, is_test=False):
    print(f"Processing {csv_path}...")
    df = pd.read_csv(csv_path)
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        qa_id = row['qa_id']
        
        vid_id = qa_id.split('_')[0] if '_' in qa_id and not qa_id.startswith('test') and not qa_id.startswith('train') else qa_id
        if len(vid_id) > 12 and qa_id.startswith('LM_test'):
            vid_id = '_'.join(qa_id.split('_')[:3])
            
        if is_test:
            vid_folder = os.path.join('videos', 'large_model_track_test', vid_id)
        else:
            vid_folder = os.path.join('videos', row['path'])
            
        for modality in MODALITIES_TO_EXTRACT:
            out_path = os.path.join(OUTPUT_DIR, f"{vid_id}_{modality}.pt")
            if os.path.exists(out_path):
                continue
                
            vid_file = os.path.join(vid_folder, modality, f"{modality}.mp4")
            if not os.path.exists(vid_file):
                # Save zero tensor if modality is missing to keep code clean later
                torch.save(torch.zeros((16, 2048)), out_path)
                continue
                
            video_tensor = get_sparse_frames(vid_file, NUM_FRAMES)
            if video_tensor is None:
                torch.save(torch.zeros((16, 2048)), out_path)
                continue
                
            video_tensor = video_tensor.to(DEVICE)
            with torch.no_grad():
                features = model(video_tensor) # [16, 2048]
                
            features = features.cpu()
            torch.save(features, out_path)

if __name__ == '__main__':
    # 1. Process Test Data
    process_dataframe('test_qa.csv', is_test=True)
    
    # 2. Process Train Data
    process_dataframe('training_qa.csv', is_test=False)
    
    print("Multi-Modal ResNet50 Feature extraction complete!")
