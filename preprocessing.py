"""
Python script for data pre-processing
"""

import os
import re
import pandas as pd

import utils.text_processing as tp
from utils.helpers import logger
from settings import RAW_DATA_DIR, COMPLEX_PROCESSED_DATA_DIR, SIMPLE_PROCESSED_DATA_DIR


def remove_referenced_name(text):
    pattern = r"@[^\s\.\?,;:!]*"
    return re.sub(pattern, " ", text).strip()


def simple_processing(text):
    text = tp.unicode_normalize(text)
    text = remove_referenced_name(text)
    text = tp.general_regex(text)
    return text


def complex_processing(text):
    text = tp.unicode_normalize(text)
    text = remove_referenced_name(text)
    text = tp.lowercasing(text)
    text = tp.general_regex(text)
    text = tp.get_decontracted_form(text)
    text = tp.keep_alpha_space(text)
    text = tp.remove_stopwords(text)
    if text != "":
        text = tp.perform_lemmatization(text)
    return text


def preprocess(data, preprocess_type="simple"):
    if preprocess_type == "simple":
        data["text"] = data["content"].map(simple_processing)
    else:
        data["text"] = data["content"].map(complex_processing)
    return data[["text", "sentiment"]]
