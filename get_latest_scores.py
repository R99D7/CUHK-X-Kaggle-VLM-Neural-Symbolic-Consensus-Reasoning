import subprocess
import io
import pandas as pd

res = subprocess.run(["kaggle", "competitions", "submissions", "-c", "cuhk-x-competition-large-model-track", "-v"], capture_output=True, text=True)
if res.returncode == 0:
    try:
        df = pd.read_csv(io.StringIO(res.stdout))
        print(df.head(15)[['fileName', 'date', 'description', 'status', 'publicScore']].to_string(index=False))
    except Exception as e:
        print(res.stdout[:2000])
else:
    print(res.stderr)
