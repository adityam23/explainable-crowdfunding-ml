import argparse
import datetime
import json
import logging
import os

import pandas as pd
import textstat

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_value(i, val):
    try:
        raw = json.loads(i)
        return raw[val]
    except Exception:
        return None


def to_iso_date(ts):
    try:
        return datetime.datetime.fromtimestamp(int(ts), datetime.UTC).strftime("%Y-%m-%d")
    except ValueError, TypeError:
        return None


def has_video(i):
    return False if str(i) == "nan" else True


def has_photo(i):
    return True if get_value(i, "key") else False


def diff_in_months(start_date, end_date):
    if not start_date or not end_date:
        return 0
    try:
        d1 = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        d2 = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        return (d2.year - d1.year) * 12 + (d2.month - d1.month)
    except ValueError:
        return 0


def get_score(i, metric):
    return metric(i)


def get_ascii(i):
    if not isinstance(i, str):
        return ""
    return i.encode("ascii", errors="ignore").decode()


def load_csv(path):
    target_columns = [
        "backers_count",
        "blurb",
        "category",
        "converted_pledged_amount",
        "country",
        "created_at",
        "deadline",
        "goal",
        "id",
        "launched_at",
        "location",
        "name",
        "photo",
        "pledged",
        "spotlight",
        "staff_pick",
        "state",
        "usd_pledged",
        "video",
        "prelaunch_activated",
        "is_liked",
        "is_disliked",
        "is_starrable",
    ]

    boolean_columns = [
        "prelaunch_activated",
        "spotlight",
        "staff_pick",
        "photo",
        "video",
        "is_liked",
        "is_disliked",
        "is_starrable",
    ]
    timestamp_columns = ["created_at", "deadline", "launched_at"]

    try:
        df = pd.read_csv(path, low_memory=False)
        # Ensure all target columns exist
        missing_cols = [col for col in target_columns if col not in df.columns]
        if missing_cols:
            logging.warning(f"Missing columns {missing_cols} in {path}. Skipping file.")
            return pd.DataFrame()

        df = df[target_columns]
        df["category"] = df["category"].apply(lambda x: get_value(x, "name"))
        df["location"] = df["location"].apply(lambda x: get_value(x, "name"))

        for col in timestamp_columns:
            df[col] = df[col].apply(to_iso_date)

        # Only successful/failed kickstarters
        df = df[(df["state"] == "successful") | (df["state"] == "failed")]
        df["state"] = df["state"].map({"successful": 1, "failed": 0})

        # Text based features
        df["blurb"] = df["blurb"].fillna("").apply(get_ascii)
        df["blurb_wc"] = df["blurb"].apply(lambda x: len(x.split()))
        df["dale_chall"] = df["blurb"].apply(
            lambda x: get_score(x, textstat.dale_chall_readability_score)
        )
        df["flesch_kincaid"] = df["blurb"].apply(
            lambda x: get_score(x, textstat.flesch_reading_ease)
        )
        df["smog"] = df["blurb"].apply(lambda x: get_score(x, textstat.smog_index))
        df["gun_fog"] = df["blurb"].apply(lambda x: get_score(x, textstat.gunning_fog))

        df["video"] = df["video"].apply(has_video)
        df["photo"] = df["photo"].apply(has_photo)

        df["camp_len"] = df.apply(
            lambda row: diff_in_months(row["launched_at"], row["deadline"]), axis=1
        )

        for col in boolean_columns:
            df[col] = df[col].astype(bool).astype(int)

        # Only keep longer blurbs
        df = df[df["blurb_wc"] >= 15]
        df = df.drop(columns=["created_at", "launched_at", "deadline"])
        return df
    except Exception as e:
        logging.error(f"Error processing {path}: {e}")
        return pd.DataFrame()


def load_all_csvs(datafolder):
    if not os.path.exists(datafolder):
        logging.error(f"Data folder {datafolder} does not exist.")
        return pd.DataFrame()

    months = [m for m in os.listdir(datafolder) if os.path.isdir(os.path.join(datafolder, m))]
    if not months:
        logging.warning("No subdirectories found in data folder.")

    dfs = []
    for month in months:
        current_month_path = os.path.join(datafolder, month)
        files = [
            f for f in os.listdir(current_month_path) if f.endswith(".csv") and "~lock" not in f
        ]
        for file in files:
            current_file_path = os.path.join(current_month_path, file)
            logging.info(f"Processing file: {current_file_path}")
            df = load_csv(current_file_path)
            if not df.empty:
                dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def main(datafolder, output_file):
    if output_file and not output_file.endswith(".csv"):
        output_file += ".csv"

    logging.info(f"Starting dataset preparation from {datafolder}")
    df = load_all_csvs(datafolder)

    if df.empty:
        logging.error("No data collected. Output file will not be created.")
        return

    df = df.dropna()
    df.to_csv(output_file, index=False)
    logging.info(f"Dataset saved to {output_file}. Processed {len(df)} records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare single dataset from the data folder.")
    parser.add_argument("-d", "--data", type=str, default="./data", help="Path to the data folder")
    parser.add_argument(
        "-f", "--file", type=str, default="full_dataset", help="Output CSV filename"
    )
    args = parser.parse_args()
    main(args.data, args.file)
