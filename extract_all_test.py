import os
import torch
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
import cv2
import pandas as pd
from tqdm import tqdm

NUM_FRAMES = 16
RESIZE_DIM = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = 'video_features_resnet'
MODALITIES = ['Depth_Color', 'Depth', 'IR']

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading ResNet50 model on {DEVICE}...")
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model = model.to(DEVICE)
model.eval()

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def get_sparse_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0: return None
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (RESIZE_DIM, RESIZE_DIM)))
        else:
            frames.append(frames[-1] if len(frames) > 0 else np.zeros((RESIZE_DIM, RESIZE_DIM, 3), dtype=np.uint8))
    cap.release()
    tensor = torch.FloatTensor(np.array(frames).transpose(0, 3, 1, 2)) / 255.0
    for i in range(3): tensor[:, i, :, :] = (tensor[:, i, :, :] - mean[i]) / std[i]
    return tensor

df = pd.read_csv('test_qa.csv')
for idx, row in tqdm(df.iterrows(), total=len(df)):
    qa_id = row['qa_id']
    
    # qa_id is like 'test_0001', we need vid_id = 'test_0001'
    vid_id = qa_id 
    # but the folder is 'LM_test_0001'
    folder_name = vid_id.replace('test_', 'LM_test_')
    
    for mod in MODALITIES:
        out_path = os.path.join(OUTPUT_DIR, f"{vid_id}_{mod}.pt")
        vid_file = os.path.join('videos', 'large_model_track_test', folder_name, mod, f"{mod}.mp4")
        
        if not os.path.exists(vid_file):
            torch.save(torch.zeros((16, 2048)), out_path)
            continue
            
        vt = get_sparse_frames(vid_file, NUM_FRAMES)
        if vt is None:
            torch.save(torch.zeros((16, 2048)), out_path)
            continue
            
        with torch.no_grad():
            features = model(vt.to(DEVICE)).cpu()
        torch.save(features, out_path)

print("Test set correctly extracted!")
