"""
- Python file to download raw data from multiple urls and saving to disk
- Combine other data in a single training file
- Then create a test set out of that, save that in the raw_data directory
- Then combine others training set with kaggle's training set
- Then divide the merged dataset into training and validation
- Save them both in raw_data directory
- And save the kaggle test set in the same directory
"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.helpers import (
    download_and_write_to_file,
    logger,
    save_csv_data,
    save_objects,
)
from utils.dataset_split import stratified_split

from settings import (
    RAW_DATA_DIR,
    ORIG_DATA_DIR,
    OTHERS_RAW_DATA,
    LUCKY_SEED,
    OBJECTS_DIR,
)


file_urls = [
    "http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/anger-ratings-0to1.train.txt",
    "http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/fear-ratings-0to1.train.txt",
    "http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/joy-ratings-0to1.train.txt",
    "http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/sadness-ratings-0to1.train.txt",
    "http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data/anger-ratings-0to1.dev.target.txt",
    "http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/fear-ratings-0to1.dev.gold.txt",
    "http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/joy-ratings-0to1.dev.gold.txt",
    "http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/sadness-ratings-0to1.dev.gold.txt",
    "http://saifmohammad.com/WebDocs/EmoInt%20Test%20Data/anger-ratings-0to1.test.target.txt",
    "http://saifmohammad.com/WebDocs/EmoInt%20Test%20Data/fear-ratings-0to1.test.target.txt",
    "http://saifmohammad.com/WebDocs/EmoInt%20Test%20Data/joy-ratings-0to1.test.target.txt",
    "http://saifmohammad.com/WebDocs/EmoInt%20Test%20Data/sadness-ratings-0to1.test.target.txt",
]


def fetch_and_save_raw_data():
    for url in file_urls:
        download_and_write_to_file(
            url, os.path.join(OTHERS_RAW_DATA, url.split("/")[-1])
        )


def merge_dataset(df_list, drop_dup_cols):
    return (
        pd.concat(df_list)
        .sample(frac=1, random_state=LUCKY_SEED)
        .dropna()
        .drop_duplicates(subset=drop_dup_cols)
        .reset_index(drop=True)
    )


def perform_label_encoding(data):
    le = LabelEncoder()
    encoded_data = le.fit_transform(data)
    return le, encoded_data


def read_all_data():
    logger.info("Creating raw training and testing dataset.")
    try:
        kaggle_data = pd.read_csv(os.path.join(ORIG_DATA_DIR, "train_data.csv"))
        anger_0 = pd.read_csv(
            os.path.join(OTHERS_RAW_DATA, "anger-ratings-0to1.train.txt"),
            sep="\t",
            header=None,
        )[[1, 2]]
        anger_1 = pd.read_csv(
            os.path.join(OTHERS_RAW_DATA, "anger-ratings-0to1.dev.target.txt"),
            sep="\t",
            header=None,
        )[[1, 2]]
        fear_0 = pd.read_csv(
            os.path.join(OTHERS_RAW_DATA, "fear-ratings-0to1.train.txt"),
            sep="\t",
            header=None,
        )[[1, 2]]
        fear_1 = pd.read_csv(
            os.path.join(OTHERS_RAW_DATA, "fear-ratings-0to1.dev.gold.txt"),
            sep="\t",
            header=None,
        )[[1, 2]]
        joy_0 = pd.read_csv(
            os.path.join(OTHERS_RAW_DATA, "joy-ratings-0to1.train.txt"),
            sep="\t",
            header=None,
        )[[1, 2]]
        joy_1 = pd.read_csv(
            os.path.join(OTHERS_RAW_DATA, "joy-ratings-0to1.dev.gold.txt"),
            sep="\t",
            header=None,
        )[[1, 2]]
        sadness_0 = pd.read_csv(
            os.path.join(OTHERS_RAW_DATA, "sadness-ratings-0to1.train.txt"),
            sep="\t",
            header=None,
        )[[1, 2]]
        sadness_1 = pd.read_csv(
            os.path.join(OTHERS_RAW_DATA, "sadness-ratings-0to1.dev.gold.txt"),
            sep="\t",
            header=None,
        )[[1, 2]]
        kaggle_test_data = pd.read_csv(os.path.join(ORIG_DATA_DIR, "test_data.csv"))
        anger_test_data = pd.read_csv(
            os.path.join(OTHERS_RAW_DATA, "anger-ratings-0to1.test.target.txt"),
            sep="\t",
            header=None,
        )[[1, 2]]
        fear_test_data = pd.read_csv(
            os.path.join(OTHERS_RAW_DATA, "fear-ratings-0to1.test.target.txt"),
            sep="\t",
            header=None,
        )[[1, 2]]
        joy_test_data = pd.read_csv(
            os.path.join(OTHERS_RAW_DATA, "joy-ratings-0to1.test.target.txt"),
            sep="\t",
            header=None,
        )[[1, 2]]
        sadness_test_data = pd.read_csv(
            os.path.join(OTHERS_RAW_DATA, "sadness-ratings-0to1.test.target.txt"),
            sep="\t",
            header=None,
        )[[1, 2]]
        other_data = merge_dataset(
            [
                anger_0,
                anger_1,
                anger_test_data,
                fear_0,
                fear_1,
                fear_test_data,
                joy_0,
                joy_1,
                joy_test_data,
                sadness_0,
                sadness_1,
                sadness_test_data,
            ],
            [1],
        )
        other_data = other_data.rename(columns={1: "content", 2: "sentiment"})
        other_data.loc[other_data["sentiment"] == "joy", "sentiment"] = "happiness"
        merged_data = merge_dataset([kaggle_data, other_data], "content")
        lbl_encoder_obj, merged_data["type"] = perform_label_encoding(
            merged_data["sentiment"]
        )
        if not save_objects(
            lbl_encoder_obj, os.path.join(OBJECTS_DIR, "label_encoder.joblib")
        ):
            logger.error("Exception in saving label encoder object!!!")
            return False
        merged_data.drop(columns=["sentiment"], inplace=True)
        train_data, test_data = stratified_split(merged_data, split_col="type")
        save_csv_data(train_data, os.path.join(RAW_DATA_DIR, "raw_train_data.csv"))
        save_csv_data(test_data, os.path.join(RAW_DATA_DIR, "raw_test_data.csv"))
        save_csv_data(
            kaggle_test_data, os.path.join(RAW_DATA_DIR, "raw_test_data_kaggle.csv")
        )
        return True
    except Exception as e:
        logger.error(
            "Exception in reading and writing all raw dataset : {}".format(str(e))
        )
        return False
