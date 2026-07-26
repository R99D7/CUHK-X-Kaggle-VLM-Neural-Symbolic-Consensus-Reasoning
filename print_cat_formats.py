import pandas as pd

te = pd.read_csv("test_qa.csv")

print("--- SEQUENCE EXAMPLES ---")
for idx, r in te[te['category'] == 'sequence'].head(3).iterrows():
    print(f"ID: {r['qa_id']} | Q: {r['question']}")
    for l in ['A', 'B', 'C', 'D']:
        print(f"  {l}: {r[l]}")

print("\n--- OBJECT_INTERACTION EXAMPLES ---")
for idx, r in te[te['category'] == 'object_interaction'].head(3).iterrows():
    print(f"ID: {r['qa_id']} | Q: {r['question']}")
    for l in ['A', 'B', 'C', 'D']:
        print(f"  {l}: {r[l]}")

print("\n--- EMOTION EXAMPLES ---")
for idx, r in te[te['category'] == 'emotion'].head(3).iterrows():
    print(f"ID: {r['qa_id']} | Q: {r['question']}")
    for l in ['A', 'B', 'C', 'D']:
        print(f"  {l}: {r[l]}")
