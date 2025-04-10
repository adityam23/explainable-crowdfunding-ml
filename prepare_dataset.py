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
    return False if str(i) == "nan" else True


def has_photo(i):
    return True if get_value(i, 'key') else False


def diff_in_months(start_date, end_date):
    # Campaign duration in months
    d1 = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    d2 = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)

def get_score(i, metric):
    return metric(i)


def get_ascii(i):
    return i.encode('ascii', errors='ignore').decode()
    

def load_csv(path):
    # print(f"Path: {path}")
    target_columns = [
        'backers_count', 'blurb', 'category', 'converted_pledged_amount', 'country',
        'created_at', 'deadline', 'goal', 'id', 'launched_at',
        'location', 'name', 'photo', 'pledged', 'spotlight', 'staff_pick', 'state',
        'usd_pledged', 'video', 'prelaunch_activated', 'is_liked', 'is_disliked', 'is_starrable'
    ]
    
    boolean_columns = ['prelaunch_activated', 'spotlight', 'staff_pick', 'photo', 'video', 'is_liked', 'is_disliked', 'is_starrable']
    
    timestamp_columns = ['created_at', 'deadline', 'launched_at']
    
    df = pd.read_csv(path, low_memory=False)
    df = df[target_columns]
    df['category'] = df['category'].apply(lambda x: get_value(x, 'name'))
    df['location'] = df['location'].apply(lambda x: get_value(x, 'name'))

    for col in timestamp_columns:
        df[col] = df[col].apply(to_iso_date)

    # Only successful/failed kickstarters
    df = df[(df["state"] == "successful") | (df["state"] == "failed")]
    df["state"] = df["state"].map({'successful': 1, 'failed': 0})
    
    # Text based features
    df["blurb"] = df["blurb"].fillna('').apply(get_ascii)
    df["blurb_wc"] = df["blurb"].apply(lambda x: len(x.split()))
    df["dale_chall"] = df["blurb"].apply(lambda x: get_score(x, textstat.dale_chall_readability_score))
    df["flesch_kincaid"] = df["blurb"].apply(lambda x: get_score(x, textstat.flesch_reading_ease))
    df["smog"] = df["blurb"].apply(lambda x: get_score(x, textstat.smog_index))
    df["gun_fog"] = df["blurb"].apply(lambda x: get_score(x, textstat.gunning_fog))
    
    df["video"] = df["video"].apply(has_video)
    df["photo"] = df["photo"].apply(has_photo)
    
    # Campaign length (in months) is more important than raw dates
    df["camp_len"] = df.apply(lambda row: diff_in_months(row["launched_at"], row["deadline"]), axis=1)
    
    # 0=False, 1=True
    for col in boolean_columns:
            df[col] = df[col].astype(bool).astype(int)
    
    # Only keep longer blurbs (around 50% of raw dataset)
    df = df[df["blurb_wc"] >= 15]
    
    df = df.drop(columns = ['created_at','launched_at','deadline'])
    return df


def load_all_csvs():
    # Base folder
    datafolder = "data"
    months = os.listdir(datafolder)
    months = [m for m in months if os.path.isdir('/'.join([datafolder,m]))]
    dfs = []
    for month in months:
        # Subfolders
        current_month = '/'.join([datafolder, month])
        files = os.listdir(current_month)
        files = [f for f in files if os.path.isfile('/'.join([current_month, f])) and "~lock" not in f]
        for file in files:
            # CSV files
            current_file = '/'.join([current_month, file])
            print(f"Processing file : {current_file}")
            df = load_csv(current_file)
            dfs.append(df)
        
    full_df = pd.concat(dfs)
    return full_df


# def tfidf(blurbs):
#     tf = TfidfVectorizer(ngram_range=(2, 2),  # bigrams only
#                             stop_words='english')  # remove common stopwords
#     tf_idf = tf.fit_transform(blurbs)
#     return pd.DataFrame(tf_idf.to_array(), columns=tf.get_feature_names_out())

FILENAME = "fullfile_ascii_3.csv"

# Load full dataset
df = load_all_csvs()
# Drop missing values, we have too much data anyway
df = df.dropna()
# Save file
df.to_csv(FILENAME)

