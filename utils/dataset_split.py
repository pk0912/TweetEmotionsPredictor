"""
Script to divide data into training, validation and testing datasets.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from settings import RAW_DATA_DIR, ORIG_DATA_DIR, TEST_DATA_DIR, LUCKY_SEED
from .helpers import logger


def read_csv_data(path, header="infer"):
    return pd.read_csv(path, header=header, encoding="utf-8")


def stratified_split(data, split_col, n_splits=1, split_ratio=0.2):
    split = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=split_ratio, random_state=LUCKY_SEED
    )
    train_set = None
    test_set = None
    try:
        for train_index, test_index in split.split(data, data[split_col]):
            train_set = data.loc[train_index]
            test_set = data.loc[test_index]
    except Exception as e:
        logger.error("Exception in Stratified split : {}".format(str(e)))
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
        os.path.join(RAW_DATA_DIR, "train.csv"), index=False, encoding="utf-8"
    )
    val_data.to_csv(
        os.path.join(RAW_DATA_DIR, "val.csv"), index=False, encoding="utf-8"
    )
    test_data.to_csv(
        os.path.join(TEST_DATA_DIR, "test.csv"), index=False, encoding="utf-8"
    )


if __name__ == "__main__":
    perform_raw_split()
