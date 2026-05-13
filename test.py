import textstat
import pandas as pd
from main import FILENAME

def safe_score(func, text):
    try:
        return func(text)
    except Exception:
        return None

df = pd.read_csv(FILENAME, low_memory=False)
df['dale_chall_score'] = df['blurb'].apply(lambda x: safe_score(textstat.dale_chall_readability_score, x))
print(df["dale_chall_score"])