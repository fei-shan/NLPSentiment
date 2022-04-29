import os
PROJECT_DIR = '.'
LOCAL_DATA_DIR = os.path.join(PROJECT_DIR, 'data')
IMDB_5K_CSV = os.path.join(LOCAL_DATA_DIR, 'IMDB Dataset.csv')
IMDB_DATA_DIR = os.path.join(LOCAL_DATA_DIR, 'acllmdb')


class Config:
    MAX_LEN = 256
    BATCH_SIZE = 32
    LR = 0.001
    VOCAB_SIZE = 30000
    EMBED_DIM = 128
    NUM_HEAD = 8  # used in bert model
    FF_DIM = 128  # used in bert model
    NUM_LAYERS = 1

# from local_config import *

