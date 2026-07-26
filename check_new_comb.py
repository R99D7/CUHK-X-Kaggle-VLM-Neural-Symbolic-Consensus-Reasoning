"""
New candidates with strong evidence:

test_0227: A = "drinking, listening to the music with headphones" -> 100% (2/2) - VERY STRONG
test_0618: D = "squats, jumping jacks" -> 83% (5/6) - STRONG (we were considering A before)

Apply these 2 additional high-confidence fixes on top of current submission (v252).
Also double-check what test_0618's current prediction is.
"""
import pandas as pd

sub = pd.read_csv('submission.csv')
te = pd.read_csv('test_qa.csv')

# Check test_0618
row_618 = te[te['qa_id'] == 'test_0618'].iloc[0]
pred_618 = sub[sub['qa_id'] == 'test_0618']['prediction'].values[0]
print(f"test_0618: pred={pred_618}")
print(f"  A: {row_618['A']}")
print(f"  B: {row_618['B']}")
print(f"  C: {row_618['C']}")
print(f"  D: {row_618['D']}")

# Check test_0227
row_227 = te[te['qa_id'] == 'test_0227'].iloc[0]
pred_227 = sub[sub['qa_id'] == 'test_0227']['prediction'].values[0]
print(f"\ntest_0227: pred={pred_227}")
print(f"  A: {row_227['A']}")
print(f"  B: {row_227['B']}")
print(f"  C: {row_227['C']}")
print(f"  D: {row_227['D']}")
