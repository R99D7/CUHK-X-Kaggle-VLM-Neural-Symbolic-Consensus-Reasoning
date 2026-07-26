"""
Qwen2-VL-2B-Instruct full inference on test videos.
Uses the locally downloaded model from Qwen2-VL-2B-Instruct-Git folder.
Uses 'path' column from test_qa.csv (not video_id).
Prefers Depth_Color modality for richer RGB signal.
Generates submission_v135_qwen2vl_full.csv
"""
import os
import gc
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = 'Qwen2-VL-2B-Instruct-Git'
VIDEO_DIR  = 'test_video'
OUTPUT_CSV = 'submission_v135_qwen2vl_full.csv'
CHECKPOINT_CSV = 'checkpoint_v135.csv'  # Save progress as we go

def extract_pred(raw_text, category):
    matches = re.findall(r'\b([A-D])\b', raw_text)
    if not matches:
        matches = re.findall(r'[A-D]', raw_text)

    if category in ['single', 'emotion', 'object_interaction', 'combination']:
        return matches[0] if matches else 'C'
    elif category == 'multi':
        unique = list(dict.fromkeys(matches))
        if not unique: return 'B'
        if len(unique) > 3: unique = unique[:3]
        return ''.join(sorted(unique))
    elif category == 'sequence':
        unique = list(dict.fromkeys(matches))
        missing = [x for x in 'ABCD' if x not in unique]
        return ''.join((unique + missing)[:4])
    return matches[0] if matches else 'C'

def build_prompt(row):
    cat = row['category']
    q   = row['question']
    opts = f"A: {row['A']}\nB: {row['B']}\nC: {row['C']}\nD: {row['D']}"

    if cat == 'sequence':
        return (f"{q}\n{opts}\n"
                "Arrange A, B, C, D in the correct order. "
                "Reply with exactly 4 letters like ABCD.")
    elif cat == 'multi':
        return (f"{q}\n{opts}\n"
                "Select 1 to 3 correct options. "
                "Reply with only the letter(s) in alphabetical order, e.g. AB or ACD.")
    else:
        return f"{q}\n{opts}\nReply with only the single correct letter A, B, C, or D."

def main():
    print(f"Loading Qwen2-VL-2B from {MODEL_PATH}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("Model loaded successfully!")

    test_df = pd.read_csv('test_qa.csv')

    # Resume from checkpoint if available
    done = set()
    if os.path.exists(CHECKPOINT_CSV):
        ckpt = pd.read_csv(CHECKPOINT_CSV)
        done = set(ckpt['qa_id'].tolist())
        results = ckpt.to_dict('records')
        print(f"Resuming from checkpoint: {len(done)} already done")
    else:
        results = []

    todo = test_df[~test_df['qa_id'].isin(done)]
    print(f"Remaining: {len(todo)} questions")

    for idx, row in tqdm(todo.iterrows(), total=len(todo)):
        qa_id = row['qa_id']
        category   = row['category']
        prompt     = build_prompt(row)

        # Build path: prefer Depth_Color (RGB-ish) over raw Depth
        raw_path = row['path']  # e.g. large_model_track_test/LM_test_0066/Depth/Depth.mp4
        base_dir = os.path.dirname(os.path.dirname(raw_path))  # large_model_track_test/LM_test_0066
        dc_path  = os.path.join(base_dir, 'Depth_Color', 'Depth_Color.mp4')
        video_path = dc_path if os.path.exists(dc_path) else raw_path

        if not os.path.exists(video_path):
            results.append({'qa_id': qa_id, 'prediction': 'C'})
            continue

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 1.5,           # Slightly more frames than before
                    "max_frames": 16,     # Caps memory usage
                },
                {"type": "text", "text": prompt},
            ],
        }]

        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)

            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            final = extract_pred(output_text, category)

        except Exception as e:
            print(f"  Error on {qa_id}: {e}")
            final = 'C'

        results.append({'qa_id': qa_id, 'prediction': final})

        # Save checkpoint every 5 questions
        if len(results) % 5 == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT_CSV, index=False)
            print(f"  Checkpoint saved: {len(results)} done")

        del inputs, generated_ids
        torch.cuda.empty_cache()

    # Final save
    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_CSV, index=False)
    pd.DataFrame(results).to_csv(CHECKPOINT_CSV, index=False)

    # Validate format
    test_merged = pd.merge(test_df, out, on='qa_id')
    errors = 0
    for cat in test_merged['category'].unique():
        cat_df = test_merged[test_merged['category'] == cat]
        preds = cat_df['prediction'].astype(str)
        lens  = preds.apply(len)
        if cat == 'sequence':
            bad = sum(lens != 4)
        elif cat in ['single', 'emotion', 'object_interaction', 'combination']:
            bad = sum(lens != 1)
        else:
            bad = sum((lens < 1) | (lens > 3))
        errors += bad
        print(f"  {cat}: min={lens.min()} max={lens.max()} bad={bad}")

    print(f"\nTotal format errors: {errors}")
    print(f"Saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
