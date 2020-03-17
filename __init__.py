import os
import sys

import settings

os.makedirs(os.path.join(settings.ROOT_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "data/orig_data"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "data/training_data"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "data/validation_data"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "data/testing_data"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "objects"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "notebooks"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "tests"), exist_ok=True)
os.makedirs(os.path.join(settings.ROOT_DIR, "docs"), exist_ok=True)

sys.path.insert(0, settings.ROOT_DIR)

