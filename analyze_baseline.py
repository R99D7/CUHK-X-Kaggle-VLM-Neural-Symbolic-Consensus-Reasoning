"""
Deep analysis of submission_ultimate_v12.csv (0.403 score)
to understand what it gets right and wrong per category.
"""
import pandas as pd
import numpy as np

test = pd.read_csv('test_qa.csv')
base = pd.read_csv('submission_ultimate_v12.csv')
train = pd.read_csv('training_qa.csv')

merged = base.merge(test[['qa_id','category','A','B','C','D','question']], on='qa_id')

print("=" * 60)
print("BASELINE 0.403 ANALYSIS")
print("=" * 60)

print("\n1. PREDICTION DISTRIBUTION PER CATEGORY:")
for cat in test['category'].unique():
    subset = merged[merged['category'] == cat]
    print(f"\n  [{cat}] n={len(subset)}")
    print("  " + str(subset['prediction'].value_counts().head(8).to_dict()))

# Compare prediction distributions to training answer distributions
print("\n\n2. TRAINING ANSWER DISTRIBUTION vs BASELINE PREDICTION DISTRIBUTION:")
for cat in ['single', 'emotion', 'combination', 'object_interaction']:
    train_sub = train[train['category'] == cat]
    base_sub = merged[merged['category'] == cat]
    
    # Training: how often is A/B/C/D the answer?
    train_dist = train_sub['answer'].value_counts(normalize=True).head(4)
    base_dist = base_sub['prediction'].value_counts(normalize=True).head(4)
    
    print(f"\n  [{cat}]")
    print(f"  Train:    A={train_dist.get('A',0):.3f} B={train_dist.get('B',0):.3f} C={train_dist.get('C',0):.3f} D={train_dist.get('D',0):.3f}")
    print(f"  Baseline: A={base_dist.get('A',0):.3f} B={base_dist.get('B',0):.3f} C={base_dist.get('C',0):.3f} D={base_dist.get('D',0):.3f}")

print("\n\n3. MULTI CATEGORY ANALYSIS:")
multi_base = merged[merged['category'] == 'multi']
multi_train = train[train['category'] == 'multi']
print(f"  Baseline multi predictions (n={len(multi_base)}):")
print("  " + str(multi_base['prediction'].value_counts().head(12).to_dict()))
print(f"\n  Training multi answers:")
multi_train_cp = multi_train.copy()
multi_train_cp['n_answers'] = multi_train_cp['answer'].str.len()
print(f"  1-answer: {(multi_train_cp['n_answers']==1).mean():.3f}")
print(f"  2-answer: {(multi_train_cp['n_answers']==2).mean():.3f}")
print(f"  3-answer: {(multi_train_cp['n_answers']==3).mean():.3f}")
print(f"  Most common multi answers: {multi_train['answer'].value_counts().head(10).to_dict()}")

print("\n\n4. SEQUENCE CATEGORY ANALYSIS:")
seq_base = merged[merged['category'] == 'sequence']
seq_train = train[train['category'] == 'sequence']
print(f"  Baseline seq predictions (n={len(seq_base)}):")
print("  " + str(seq_base['prediction'].value_counts().head(10).to_dict()))
print(f"  Training seq top answers:")
print("  " + str(seq_train['answer'].value_counts().head(10).to_dict()))

print("\n\n5. UNIQUE QUESTIONS ANALYSIS:")
for cat in ['single', 'emotion', 'combination']:
    subset = test[test['category']==cat]
    unique_qs = subset['question'].nunique()
    print(f"  {cat}: {unique_qs} unique questions out of {len(subset)} total")
