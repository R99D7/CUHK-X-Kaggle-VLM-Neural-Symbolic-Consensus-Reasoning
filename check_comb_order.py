"""
Check if combination options distinguish order.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
comb = tr[tr['category'] == 'combination']

order_matters = False
for idx, row in comb.iterrows():
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    opt_sets = [frozenset(txt.split(',')) for txt in opts.values()]
    # If the number of unique sets is less than 4, it means some options have the same actions in different order!
    if len(set(opt_sets)) < len(opt_sets):
        print(f"Order matters for {row['qa_id']}!")
        order_matters = True

if not order_matters:
    print("Order does NOT matter for combination questions. They are just sets.")
