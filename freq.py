import pandas as pd
df = pd.read_csv('training_qa.csv')
multi = df['category'] == 'multi'
for l in [1, 2, 3, 4]:
    mask = multi & (df['answer'].astype(str).map(len) == l)
    print(f"Len {l}:", df[mask]['answer'].value_counts().head(3).to_dict())
