import os
import cv2
import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

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
    print("Starting Zero-Shot Inference with Florence-2-large...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_id = "microsoft/Florence-2-large"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    base_dir = r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK"
    test_csv = os.path.join(base_dir, "test_qa.csv")
    
    df = pd.read_csv(test_csv)
    # Process full test set
    target_df = df
    
    predictions = []
    
    for idx, row in target_df.iterrows():
        qa_id = row['qa_id']
        vid_path = os.path.join(base_dir, "videos", row['path'])
        
        # Fallbacks for paths
        if not os.path.exists(vid_path):
            vid_path = os.path.join(base_dir, "test_video", row['path'])
        if not os.path.exists(vid_path):
            vid_path = os.path.join(base_dir, row['path'])
        if not os.path.exists(vid_path):
            vid_path = os.path.join(base_dir, row['path'].replace('large_model_track_test/', 'test_video/large_model_track_test/'))
            
        default_pred = 'ABCD' if row['category'] == 'sequence' else 'AB' if row['category'] == 'multi' else 'A'
        
        if not os.path.exists(vid_path):
            predictions.append({"qa_id": qa_id, "prediction": default_pred})
            continue
            
        try:
            grid_img = stitch_grid(vid_path)
            if grid_img is None:
                predictions.append({"qa_id": qa_id, "prediction": default_pred})
                continue
                
            q_text = f"Question: {row['question']}\nA) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}\nOutput only the correct option letter(s)."
            prompt = f"<QA> {q_text}"
            
            inputs = processor(text=prompt, images=grid_img, return_tensors="pt").to(device, torch.float16)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=10,
                    num_beams=3
                )
                
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(
                generated_text, 
                task="<QA>", 
                image_size=(grid_img.width, grid_img.height)
            )
            
            ans = parsed_answer.get("<QA>", "")
            
            # Clean up answer
            pred = ''.join([c for c in ans.upper() if c in 'ABCD'])
            
            if row['category'] == 'sequence':
                pred = ''.join(dict.fromkeys(pred))
                missing = [c for c in 'ABCD' if c not in pred]
                pred = pred + ''.join(missing)
                if len(pred) != 4 or sorted(pred) != ['A', 'B', 'C', 'D']:
                    pred = 'ABCD'
            elif row['category'] == 'multi':
                pred = ''.join(sorted(dict.fromkeys(pred)))
                if len(pred) == 0:
                    pred = 'AB'
            else:
                if len(pred) == 0: pred = 'A'
                pred = pred[0]
                
            print(f"{qa_id}: {pred} (Raw: {ans})")
            predictions.append({"qa_id": qa_id, "prediction": pred})
            
        except Exception as e:
            print(f"Error on {qa_id}: {e}")
            predictions.append({"qa_id": qa_id, "prediction": default_pred})
            
    df_pred = pd.DataFrame(predictions)
    
    # Blend with v151 for missing ones
    df_base = pd.read_csv(os.path.join(base_dir, 'submission_v151_v143_bugfix.csv'))
    df_pred_dict = dict(zip(df_pred['qa_id'], df_pred['prediction']))
    
    df_base['prediction'] = df_base.apply(
        lambda row: df_pred_dict.get(row['qa_id'], row['prediction']),
        axis=1
    )
    
    out_path = os.path.join(base_dir, 'submission_florence2_zeroshot.csv')
    df_base.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
