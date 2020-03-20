"""
Python file containing methods that are common requirement throughout the project
"""

from urllib import request
import pandas as pd
from joblib import dump


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def download_and_write_to_file(url, filepath):
    response = request.urlopen(url)
    content = response.read()
    with open(filepath, "wb") as f:
        f.write(content)


def read_csv_data(path, keep_columns=[], drop_cols=[]):
    data = pd.read_csv(path, encoding="utf-8")
    if len(keep_columns) > 0:
        data = data[keep_columns]
    data = data.dropna(subset=drop_cols, how="all")
    data = data.sample(frac=1, random_state=LUCKY_SEED).reset_index(drop=True)
    return data


def save_objects(obj, path):
    try:
        dump(obj, path)
    except Exception as e:
        print("ERROR IN SAVING OBJECT : ", e)
        return False
    return True
