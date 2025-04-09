import pandas as pd
import json
import os
import datetime
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer

def get_value(i, val):
    try:
        raw = json.loads(i)
        return raw[val]
    except Exception:
        return None


def to_iso_date(ts):
    return datetime.datetime.fromtimestamp(int(ts), datetime.UTC).strftime('%Y-%m-%d')


def has_video(i):
    return True if get_value(i, 'status') == "successful" else False


def has_photo(i):
    return True if get_value(i, 'key') else False


def diff_in_months(start_date, end_date):
    d1 = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    d2 = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)

def get_score(i, metric):
    return metric(i)


def get_ascii(i):
    return str(i.encode('ascii', errors='ignore'))
    

def load_csv(path):
    # print(f"Path: {path}")
    target_columns = [
        'backers_count', 'blurb', 'category', 'converted_pledged_amount', 'country',
        'created_at', 'deadline', 'disable_communication', 'goal', 'id', 'launched_at',
        'location', 'name', 'photo', 'pledged', 'spotlight', 'staff_pick', 'state',
        'usd_pledged', 'video'
    ]
    
    boolean_columns = ['disable_communication', 'spotlight', 'staff_pick']
    
    timestamp_columns = ['created_at', 'deadline', 'launched_at']
    
    df = pd.read_csv(path, low_memory=False)
    df = df[target_columns]
    df['category'] = df['category'].apply(lambda x: get_value(x, 'name'))
    df['location'] = df['location'].apply(lambda x: get_value(x, 'name'))

    for col in timestamp_columns:
        df[col] = df[col].apply(to_iso_date)

    for col in boolean_columns:
            df[col] = df[col].astype(bool)
    
    df["blurb"] = df["blurb"].fillna('').apply(get_ascii)
    df["blurb_wc"] = df["blurb"].apply(lambda x: len(x.split()))
    df["dale_chall"] = df["blurb"].apply(lambda x: get_score(x, textstat.dale_chall_readability_score))
    df["flesch_kincaid"] = df["blurb"].apply(lambda x: get_score(x, textstat.flesch_reading_ease))
    df["smog"] = df["blurb"].apply(lambda x: get_score(x, textstat.smog_index))
    df["gun_fog"] = df["blurb"].apply(lambda x: get_score(x, textstat.gunning_fog))
    
    df["video"] = df["video"].apply(has_video)
    df["photo"] = df["photo"].apply(has_photo)
    df["camp_len"] = df.apply(lambda row: diff_in_months(row["launched_at"], row["deadline"]), axis=1)
    
    # df = df[df["blurb_wc"] >= 10].drop(columns=["blurb_wc"])
    return df


def load_all_csvs():
    datafolder = "data"
    months = os.listdir(datafolder)
    months = [m for m in months if os.path.isdir('/'.join([datafolder,m]))]
    dfs = []
    for month in months:
        current_month = '/'.join([datafolder, month])
        files = os.listdir(current_month)
        files = [f for f in files if os.path.isfile('/'.join([current_month, f])) and "~lock" not in f]
        for file in files:
            current_file = '/'.join([current_month, file])
            df = load_csv(current_file)
            dfs.append(df)
        
    full_df = pd.concat(dfs)
    return full_df


def tfidf(blurbs):
    tf = TfidfVectorizer(ngram_range=(2, 2),  # bigrams only
                            stop_words='english')  # remove common stopwords
    return tf.fit_transform(blurbs)

FILENAME = "fullfile_ascii_1.csv"



df = load_all_csvs()
print(df.info())
print(df.describe())
df = df.dropna()
df.to_csv(FILENAME)

tf_blurb = tfidf(df["blurbs"])
print(tf_blurb)


# print(glob(month, '*.csv'))
# print(os.listdir(month))
# df = pd.read_csv("data/Kickstarter_2020-01-16T03_20_15_556Z/Kickstarter.csv")
# print(df.columns)

# df = pd.read_csv(FILENAME, low_memory=False)
# df.dropna()
# df.dropna(axis=1)
# df.to_csv(f"dropped+{FILENAME}")
# print(df.columns)
# print(df["location"])
# # df = df[df["blurb_wc"] >= 50].drop(columns=["blurb_wc"])
# # print(df.head())


