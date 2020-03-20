"""
Script to divide data into training, validation and testing datasets.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

DATA_DIR = "data/"

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data/")
ORIG_DATA_DIR = os.path.join(RAW_DATA_DIR, "sa-emotions")
TRAIN_DIR = os.path.join(DATA_DIR, "training_data")
VALID_DIR = os.path.join(DATA_DIR, "validation_data")
TEST_DIR = os.path.join(DATA_DIR, "testing_data")

LUCKY_SEED = 42
TRAIN_FILE_COUNT = 43
VAL_FILE_COUNT = 12


def read_csv_data(path):
    return pd.read_csv(path)


def stratified_split(data, split_col, n_splits=1, split_ratio=0.2):
    split = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=split_ratio, random_state=LUCKY_SEED
    )
    for train_index, test_index in split.split(data, data[split_col]):
        train_set = data.loc[train_index]
        test_set = data.loc[test_index]
    return train_set, test_set


def save_to_multiple_csv_files(data, save_path, name_prefix, header, n_parts=10):
    path_format = os.path.join(save_path, "{}_{:02d}.csv")
    filepaths = []
    m = len(data)
    for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filepaths.append(part_csv)
        with open(part_csv, "wt", encoding="utf-8") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([repr(col) for col in data[row_idx]]))
                f.write("\n")
    return filepaths


def perform_raw_split():
    data = read_csv_data(os.path.join(ORIG_DATA_DIR, "train_data.csv"))
    data = data.drop_duplicates(subset="content").reset_index(drop=True)
    data = data.sample(frac=1, random_state=LUCKY_SEED).reset_index(drop=True)
    train_data, val_data = stratified_split(data, split_col="sentiment")
    test_data = read_csv_data(os.path.join(ORIG_DATA_DIR, "test_data.csv"))
    train_data.to_csv(
        os.path.join(RAW_DATA_DIR, "train.csv"), index=None, encoding="utf-8"
    )
    val_data.to_csv(os.path.join(RAW_DATA_DIR, "val.csv"), index=None, encoding="utf-8")
    test_data.to_csv(os.path.join(TEST_DIR, "test.csv"), index=None, encoding="utf-8")


if __name__ == "__main__":
    perform_raw_split()
