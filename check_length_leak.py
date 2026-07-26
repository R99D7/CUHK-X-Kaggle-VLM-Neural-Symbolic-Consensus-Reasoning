"""
Check option length leak in training data.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
total = 0
longest_is_correct = 0

for idx, row in tr.iterrows():
    ans = str(row['answer'])
    if len(ans) > 1: continue # only check single choice
    
    opts = {l: len(str(row[l]).strip()) for l in ['A', 'B', 'C', 'D']}
    longest = max(opts, key=opts.get)
    
    # What if there's a tie?
    max_len = opts[longest]
    longest_opts = [l for l, v in opts.items() if v == max_len]
    
    if ans in longest_opts:
        # If we guess the longest (or one of the longest), do we get it right?
        # To be fair, let's say we get 1/len(longest_opts) correct
        longest_is_correct += 1.0 / len(longest_opts)
        
    total += 1

print(f"Longest option correct: {longest_is_correct:.2f} / {total} ({longest_is_correct/total:.2%})")
print("Random chance would be 25.00%")
