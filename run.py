"""
Entry point for training and predicting
"""
import os
import pandas as pd
from utils.helpers import logger
from raw_data_ingest_merge import fetch_and_save_raw_data, read_all_data
from preprocessing import preprocess
from config import DOWNLOAD_RAW_DATA, CREATE_RAW_DATASET, SIMPLE_PROCESSING_TYPE, COMPLEX_PROCESSING_TYPE
from settings import RAW_DATA_DIR, SIMPLE_PROCESSED_DATA_DIR, COMPLEX_PROCESSED_DATA_DIR


def main():
    logger.info("Execution Started!!!")
    if DOWNLOAD_RAW_DATA:
        fetch_and_save_raw_data()
    if CREATE_RAW_DATASET:
        read_all_data()
    train_data = pd.read_csv(
        os.path.join(RAW_DATA_DIR, "raw_train_data.csv"), encoding="utf-8"
    )
    val_data = pd.read_csv(
        os.path.join(RAW_DATA_DIR, "raw_val_data.csv"), encoding="utf-8"
    )
    if SIMPLE_PROCESSING_TYPE:
        logger.info("Performing simple text processing.")
        train_data_simple = preprocess(train_data)
        val_data_simple = preprocess(val_data)
        train_data_simple.to_csv(
            os.path.join(SIMPLE_PROCESSED_DATA_DIR, "train_data_simple.csv"),
            index=False,
            encoding="utf-8",
        )
        val_data_simple.to_csv(
            os.path.join(SIMPLE_PROCESSED_DATA_DIR, "val_data_simple.csv"),
            index=False,
            encoding="utf-8",
        )
    if COMPLEX_PROCESSING_TYPE:
        logger.info("Performing complex text processing.")
        train_data_complex = preprocess(train_data, preprocess_type="complex")
        val_data_complex = preprocess(val_data, preprocess_type="complex")
        train_data_complex.to_csv(
            os.path.join(COMPLEX_PROCESSED_DATA_DIR, "train_data_complex.csv"),
            index=False,
            encoding="utf-8",
        )
        val_data_complex.to_csv(
            os.path.join(COMPLEX_PROCESSED_DATA_DIR, "val_data_complex.csv"),
            index=False,
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
