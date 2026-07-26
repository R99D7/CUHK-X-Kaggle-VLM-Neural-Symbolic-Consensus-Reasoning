import os
import glob
import pandas as pd

files = sorted(glob.glob("submission_v*.csv") + ["submission_global.csv", "submission_with_pseudo.csv", "submission.csv"])
data = {}

for f in set(files):
    if os.path.exists(f):
        try:
            df = pd.read_csv(f)
            if len(df) == 682 and 'prediction' in df.columns:
                data[f] = dict(zip(df['qa_id'], df['prediction']))
        except Exception as e:
            pass

base_name = 'submission_v265_MULTI2COMB.csv'
ref = data[base_name]

print(f"--- High-Similarity Files (Matching {base_name} >= 80%) ---")
for fname, preds in sorted(data.items(), key=lambda x: -os.stat(x[0]).st_mtime):
    matches = sum(1 for k, v in ref.items() if str(v).strip().upper() == str(preds.get(k, '')).strip().upper())
    ratio = matches / 682
    if ratio >= 0.80 or "v26" in fname or "v27" in fname or fname == "submission.csv":
        mtime = os.stat(fname).st_mtime
        import datetime
        dt = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{dt}] {fname:35s} -> matches v265: {matches:3d}/682 ({ratio:.2%})")
