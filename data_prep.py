import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import json

def stitch_grid(vid_path, target_size=(336, 336)):
    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return None
        
    frame_indices = [
        int(total_frames * 0.1),
        int(total_frames * 0.35),
        int(total_frames * 0.65),
        int(total_frames * 0.9)
    ]
    
    frames = []
    for f_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(f_idx, total_frames - 1))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (target_size[0]//2, target_size[1]//2))
            frames.append(frame)
            
    cap.release()
    
    if len(frames) == 0:
        return None
        
    while len(frames) < 4:
        frames.append(frames[-1])
        
    top_row = np.hstack((frames[0], frames[1]))
    bottom_row = np.hstack((frames[2], frames[3]))
    grid_img = np.vstack((top_row, bottom_row))
    return Image.fromarray(grid_img)

def main():
    print("Starting data preparation for Florence-2 Fine-Tuning...")
    base_dir = r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK"
    train_csv = os.path.join(base_dir, "training_qa.csv")
    
    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found!")
        return
        
    df = pd.read_csv(train_csv)
    # The user asked to train/test/validate. We can use the whole training_qa for train/val
    
    out_dir = os.path.join(base_dir, "training_grids")
    os.makedirs(out_dir, exist_ok=True)
    
    dataset = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        qa_id = row['qa_id']
        vid_path = os.path.join(base_dir, "videos", row['path'])
        
        # Fallbacks for paths
        if not os.path.exists(vid_path):
            vid_path = os.path.join(base_dir, row['path'])
            
        if not os.path.exists(vid_path):
            continue
            
        out_img_path = os.path.join(out_dir, f"{qa_id}.jpg")
        
        if not os.path.exists(out_img_path):
            grid_img = stitch_grid(vid_path)
            if grid_img is None:
                continue
            grid_img.save(out_img_path)
            
        # Format for Florence-2 VQA
        # Prompt structure: "<vqa> Question: {q} A) {a} B) {b} C) {c} D) {d}"
        q_text = f"Question: {row['question']}\nA) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}"
        ans = row['answer']
        
        dataset.append({
            "image": out_img_path,
            "text": q_text,
            "answer": ans
        })
        
    # Split into 90% train, 10% val
    np.random.seed(42)
    np.random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.9)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    
    with open(os.path.join(base_dir, "florence2_train.json"), "w") as f:
        json.dump(train_data, f, indent=4)
    with open(os.path.join(base_dir, "florence2_val.json"), "w") as f:
        json.dump(val_data, f, indent=4)
        
    print(f"Data prep complete! Train: {len(train_data)}, Val: {len(val_data)}")

if __name__ == "__main__":
    main()
