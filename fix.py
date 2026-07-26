with open('train_v32_multimodal_cv.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if 'if fold == 1:' in line and i+1 < len(lines) and 'Stopping after Fold 2' in lines[i+1]:
        skip = True
        continue
    if skip and 'break' in line:
        skip = False
        continue
    if skip:
        continue
    
    if 'oof_pred_df = pd.concat([oof_pred_df, pd.DataFrame(fold_res)], ignore_index=True)' in line:
        new_lines.append(line)
        new_lines.append('        if fold == 1:\n')
        new_lines.append('            print("Stopping after Fold 2 (2-Fold mode) to save time...")\n')
        new_lines.append('            break\n')
    else:
        new_lines.append(line)

with open('train_v32_multimodal_cv.py', 'w') as f:
    f.writelines(new_lines)
