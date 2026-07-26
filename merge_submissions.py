import pandas as pd

def merge_submissions(part1_path='submission_part1.csv', part2_path='submission_part2.csv', out_path='submission_qwen2_final.csv'):
    print(f"Loading {part1_path} and {part2_path}...")
    try:
        df1 = pd.read_csv(part1_path)
        df2 = pd.read_csv(part2_path)
        
        # Merge them (Part 1 has first 341, Part 2 has rest)
        merged = pd.concat([df1, df2]).drop_duplicates(subset='qa_id', keep='first')
        
        merged.to_csv(out_path, index=False)
        print(f"Successfully merged! Output saved to {out_path}")
        print(f"Total predictions: {len(merged)}")
    except Exception as e:
        print(f"Error merging files: {e}")
        
if __name__ == "__main__":
    merge_submissions()
