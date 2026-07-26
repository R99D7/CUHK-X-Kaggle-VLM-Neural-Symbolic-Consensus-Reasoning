import pandas as pd
df = pd.read_csv('submission_v200_GEMINI_VISION_ULTIMATE.csv')
print(df['prediction'].value_counts())
