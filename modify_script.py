import re

with open('train_ml_v11.py', 'r') as f:
    code = f.read()

save_code = """
    print("Saving raw probabilities...")
    prob_df = []
    for qa_id, probs_for_this_id in predictions.items():
        probs_for_this_id.sort(key=lambda x: x[1])
        prob_dict = {'qa_id': qa_id}
        for prob, letter in probs_for_this_id:
            prob_dict[f'prob_{letter}'] = prob
        prob_df.append(prob_dict)
    
    pd.DataFrame(prob_df).to_csv('v11_raw_probs.csv', index=False)
"""

code = code.replace('print("Saving submission_ml_v11.csv...")', save_code + '\n    print("Saving submission_ml_v11.csv...")')

with open('train_ml_v11_save_probs.py', 'w') as f:
    f.write(code)

print('Modified script saved.')
