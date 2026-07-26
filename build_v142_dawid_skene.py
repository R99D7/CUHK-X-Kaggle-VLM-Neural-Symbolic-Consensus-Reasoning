import pandas as pd
import numpy as np
import glob

print("Loading top submissions...")

# We select a diverse pool of our best models (avoiding the failed exact leaks)
target_files = [
    'submission_v117_ultimate_multimodal.csv',
    'submission_v116_ultimate_safe_threshold_040.csv',
    'submission_v114_ultimate_safe_dual_agreement.csv',
    'submission_v99_top_kaggle_guarantee.csv',
    'submission_v134_soft_prob_ensemble.csv',
    'submission_v65_surgical_strike.csv',
    'submission_v60_mega_forest.csv',
    'submission_v51_pseudo_label.csv',
    'submission_v106_mega_ensemble.csv',
    'submission_v136_qwen2vl_v117_blend.csv' # Including a weaker model so EM can learn to downweight it
]

dfs = {}
for f in target_files:
    try:
        df = pd.read_csv(f).set_index('qa_id')
        dfs[f] = df['prediction'].astype(str).str.strip()
    except Exception as e:
        print(f"Skipping {f}: {e}")

# Build annotation matrix Y (questions x models)
df_all = pd.DataFrame(dfs)
qa_ids = df_all.index.tolist()
models = df_all.columns.tolist()

num_items = len(qa_ids)
num_models = len(models)

# Map string predictions to integer classes
unique_classes = sorted(list(set(df_all.values.flatten())))
num_classes = len(unique_classes)
class_to_idx = {c: i for i, c in enumerate(unique_classes)}
idx_to_class = {i: c for i, c in enumerate(unique_classes)}

Y = np.zeros((num_items, num_models), dtype=int)
for i, qa_id in enumerate(qa_ids):
    for j, model in enumerate(models):
        pred = df_all.iloc[i, j]
        Y[i, j] = class_to_idx[pred]

print(f"Data shape: {num_items} questions, {num_models} models, {num_classes} unique classes.")

# --- Dawid-Skene EM Algorithm ---

# Initialize true label probabilities via Majority Vote
T = np.zeros((num_items, num_classes))
for i in range(num_items):
    counts = np.bincount(Y[i, :], minlength=num_classes)
    T[i, :] = counts / num_models

class_marginals = np.zeros(num_classes)
confusion_matrices = np.zeros((num_models, num_classes, num_classes))

MAX_ITER = 30
TOL = 1e-4

print("Starting Expectation-Maximization...")
for iteration in range(MAX_ITER):
    # M-STEP: Estimate confusion matrices and class marginals given T
    class_marginals = np.mean(T, axis=0) + 1e-6 # Add smoothing
    class_marginals /= np.sum(class_marginals)
    
    for k in range(num_models):
        for j in range(num_classes): # True class
            for l in range(num_classes): # Predicted class
                # Sum of probabilities that item i is class j, given that model k predicted l
                mask = (Y[:, k] == l)
                confusion_matrices[k, j, l] = np.sum(T[mask, j]) + 1e-6 # Laplace smoothing
            confusion_matrices[k, j, :] /= np.sum(confusion_matrices[k, j, :])
            
    # E-STEP: Estimate true labels given confusion matrices and class marginals
    old_T = np.copy(T)
    for i in range(num_items):
        for j in range(num_classes):
            prob = np.log(class_marginals[j])
            for k in range(num_models):
                pred_class = Y[i, k]
                prob += np.log(confusion_matrices[k, j, pred_class])
            T[i, j] = prob
            
        # Log-sum-exp trick for numerical stability
        max_prob = np.max(T[i, :])
        T[i, :] = np.exp(T[i, :] - max_prob)
        T[i, :] /= np.sum(T[i, :])
        
    diff = np.sum(np.abs(T - old_T))
    if diff < TOL:
        print(f"Converged at iteration {iteration}")
        break

# --- Evaluate learned model accuracies ---
print("\nLearned Model Accuracies (diagonal of confusion matrix average):")
for k, model in enumerate(models):
    acc = np.mean([confusion_matrices[k, j, j] for j in range(num_classes)])
    print(f"{model}: {acc:.4f}")

# --- Generate Final Predictions ---
final_preds = []
changed_from_mv = 0

for i, qa_id in enumerate(qa_ids):
    best_class_idx = np.argmax(T[i, :])
    best_class = idx_to_class[best_class_idx]
    
    # Compare with simple majority vote
    mv_counts = np.bincount(Y[i, :])
    mv_class = idx_to_class[np.argmax(mv_counts)]
    
    if best_class != mv_class:
        changed_from_mv += 1
        
    final_preds.append({'qa_id': qa_id, 'prediction': best_class})

out_df = pd.DataFrame(final_preds)
out_df.to_csv('submission_v142_dawid_skene.csv', index=False)

print(f"\nSaved submission_v142_dawid_skene.csv!")
print(f"The Dawid-Skene ensemble diverged from naive Majority Vote on {changed_from_mv} questions.")
