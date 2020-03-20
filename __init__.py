import os
import sys

import settings

os.makedirs(os.path.join(settings.ROOT_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "objects"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "notebooks"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "tests"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "docs"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "data/raw_data"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "data/processed_data"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "data/training_data"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "data/validation_data"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "data/testing_data"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "data/raw_data/others"), exist_ok=True)

# For text based data
os.makedirs(
    os.path.join(settings.ROOT_DIR, "data/processed_data/simple"), exist_ok=True
)
os.makedirs(
    os.path.join(settings.ROOT_DIR, "data/processed_data/complex"), exist_ok=True
)
os.makedirs(
    os.path.join(settings.ROOT_DIR, "data/processed_data/simple/training_data"),
    exist_ok=True,
)
os.makedirs(
    os.path.join(settings.ROOT_DIR, "data/processed_data/simple/validation_data"),
    exist_ok=True,
)
os.makedirs(
    os.path.join(settings.ROOT_DIR, "data/processed_data/complex/training_data"),
    exist_ok=True,
)
os.makedirs(
    os.path.join(settings.ROOT_DIR, "data/processed_data/complex/validation_data"),
    exist_ok=True,
)
os.makedirs(
    os.path.join(settings.ROOT_DIR, "data/training_data/one_hot_encoded"), exist_ok=True
)
os.makedirs(
    os.path.join(settings.ROOT_DIR, "data/validation_data/one_hot_encoded"),
    exist_ok=True,
)
os.makedirs(os.path.join(settings.ROOT_DIR, "data/training_data/tf_idf"), exist_ok=True)
os.makedirs(
    os.path.join(settings.ROOT_DIR, "data/validation_data/tf_idf"), exist_ok=True
)

sys.path.insert(0, settings.ROOT_DIR)
