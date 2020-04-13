"""
Contains configurations which will get executed
"""

DOWNLOAD_RAW_DATA = False
CREATE_RAW_DATASET = True
SIMPLE_PROCESSING_TYPE = True
COMPLEX_PROCESSING_TYPE = True
VECTORIZE_DATA_SIMPLE = False
VECTORIZE_DATA_COMPLEX = True
SIMPLE_DATA_CLASSIFICATION = False
COMPLEX_DATA_CLASSIFICATION = False
TEST_SET_EVALUATION = False
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 64
EPOCHS = 200
EARLY_STOP_PATIENCE = 10
LOSS = "sparse_categorical_crossentropy"
METRICS = "accuracy"
MODEL_NAME = "Model_"
MODEL_SAVE_FORMAT = ".h5"
GLOVE_PATH = "/Users/praveenkumar/Own_works/resources/nlp/glove"
CLASSES = ["type"]
PREDICTION_DATA_TYPE = "complex"
THRESHOLD = 0.7
