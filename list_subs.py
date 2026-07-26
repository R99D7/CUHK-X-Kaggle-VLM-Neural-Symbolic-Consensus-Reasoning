import os
import glob
import pandas as pd

files = glob.glob("submission_v*.csv") + glob.glob("submission_*.csv")
data = []
for f in files:
    stat = os.stat(f)
    data.append({
        'file': f,
        'mtime': stat.st_mtime,
        'size': stat.st_size
    })
    
df = pd.DataFrame(data)
df = df.sort_values(by='mtime', ascending=False)
for idx, r in df.head(30).iterrows():
    import datetime
    dt = datetime.datetime.fromtimestamp(r['mtime']).strftime('%Y-%m-%d %H:%M:%S')
    print(f"{dt} | {r['size']:8d} | {r['file']}")
