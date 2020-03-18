"""
Python file to download raw data from multiple urls and saving to disk
"""

import os
from utils import download_and_write_to_file

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
    "http://saifmohammad.com/WebDocs/EmoInt%20Test%20Data/sadness-ratings-0to1.test.target.txt"
]

DATA_DIR = "data/"

RAW_DATA_DIR = os.path.join(DATA_DIR, "orig_data")
OTHERS_RAW_DATA = os.path.join(RAW_DATA_DIR, "others")


def fetch_and_save_raw_data():
    for url in file_urls:
        download_and_write_to_file(url, os.path.join(OTHERS_RAW_DATA, url.split("/")[-1]))


if __name__ == "__main__":
    fetch_and_save_raw_data()
