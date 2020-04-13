"""
Entry point for training and predicting
"""
import os
import numpy as np
import pandas as pd
from joblib import dump, load
from utils.helpers import logger, save_csv_data
from utils.dataset_split import stratified_split
from raw_data_ingest_merge import fetch_and_save_raw_data, read_all_data
from preprocessing import preprocess
from vectorizer import vectorize
from config import (
    DOWNLOAD_RAW_DATA,
    CREATE_RAW_DATASET,
    SIMPLE_PROCESSING_TYPE,
    COMPLEX_PROCESSING_TYPE,
    VECTORIZE_DATA_SIMPLE,
    VECTORIZE_DATA_COMPLEX,
    MAX_VOCAB_SIZE,
    MAX_SEQUENCE_LENGTH,
)
from settings import (
    RAW_DATA_DIR,
    SIMPLE_PROCESSED_DATA_DIR,
    COMPLEX_PROCESSED_DATA_DIR,
    TRAIN_DATA_DIR_WI,
    VAL_DATA_DIR_WI,
    OBJECTS_DIR,
)


def vectorize_data(data_path, processing_type):
    data = pd.read_csv(data_path, encoding="utf-8")
    tokenizer, vectors = vectorize(
        MAX_VOCAB_SIZE, MAX_SEQUENCE_LENGTH, data["text"].values
    )
    if tokenizer is not None and vectors is not None:
        dump(tokenizer, os.path.join(OBJECTS_DIR, "tokenizer.joblib"))
        vec_df = pd.DataFrame(vectors)
        vec_df = pd.concat([vec_df, data.drop(columns=["text"])], axis=1)
        train_vec_data, val_vec_data = stratified_split(vec_df, split_col="type")
        save_csv_data(
            train_vec_data,
            os.path.join(
                TRAIN_DATA_DIR_WI, "train_vectors_{}.csv".format(processing_type)
            ),
        )
        save_csv_data(
            val_vec_data,
            os.path.join(VAL_DATA_DIR_WI, "val_vectors_{}.csv".format(processing_type)),
        )
    else:
        logger.error("Error in vectorizing data!!!")
        return False
    return True


def main():
    logger.info("Execution Started!!!")
    if DOWNLOAD_RAW_DATA:
        fetch_and_save_raw_data()
    if CREATE_RAW_DATASET:
        if not read_all_data():
            logger.error("Execution abruptly stopped while creating raw dataset!!!")
            return
    try:
        train_data = pd.read_csv(
            os.path.join(RAW_DATA_DIR, "raw_train_data.csv"), encoding="utf-8"
        )
        if SIMPLE_PROCESSING_TYPE:
            logger.info("Performing simple text processing.")
            train_data_simple = preprocess(train_data)
            if type(train_data_simple) == pd.core.frame.DataFrame:
                train_data_simple.to_csv(
                    os.path.join(SIMPLE_PROCESSED_DATA_DIR, "train_data_simple.csv"),
                    index=False,
                    encoding="utf-8",
                )
            else:
                logger.error("Unable to write simple processed data!!!")
                return
        if COMPLEX_PROCESSING_TYPE:
            logger.info("Performing complex text processing.")
            train_data_complex = preprocess(train_data, preprocess_type="complex")
            if type(train_data_complex) == pd.core.frame.DataFrame:
                train_data_complex.to_csv(
                    os.path.join(COMPLEX_PROCESSED_DATA_DIR, "train_data_complex.csv"),
                    index=False,
                    encoding="utf-8",
                )
            else:
                logger.error("Unable to write complex processed data!!!")
                return
        if VECTORIZE_DATA_SIMPLE:
            logger.info("Vectorizing simple processed data.")
        if VECTORIZE_DATA_COMPLEX:
            logger.info("Vectorizing complex processed data.")
            if not vectorize_data(
                os.path.join(COMPLEX_PROCESSED_DATA_DIR, "train_data_complex.csv"),
                "complex",
            ):
                logger.error(
                    "Execution abruptly stopped while vectorizing complex data!!!"
                )
                return
    except Exception as e:
        logger.error("Exception in main method : {}".format(str(e)))
        return
    logger.info("Execution successfully completed.")


if __name__ == "__main__":
    main()
