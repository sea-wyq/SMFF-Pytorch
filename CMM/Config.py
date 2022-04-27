"""
@author:32369
@file:Config.py
@time:2021/12/03
"""


class Config:
    DATA_PATH = "../"

    HISTORY_MAX_LENGTH = 50
    FEATURE_MAX_LENGTH = 15

    HISTORY_MIN_LENGTH = 200
    nsample = 4
    EPOCHS = 10
    BATCH_SIZE = 32
    LR = 5e-4
    SHUFFLE = True
    NUM_WORKERS = 0
    DROUPT = 0.2

    DATASET_TYPE = "small"
    PRINT_LENGHT = 2000

    BEHAVIORS_NAME = ["ImpressionID", "UserID", "Time", "History", "Impressions"]
    NEWS_NAME = ["NewsID", "Category", "SubCategory", "Title", "Abstract", "Body_url", "TitleEntities",
                 "AbstractEntities"]

    PROCESS_DATA_PATH = "../processedData/"
    GLOVE_PATH = 'glove/glove.6B.300d.txt'
    FEATURE_EMB_LENGTH = 300
    Q, K, V = 300, 512, 512
    HIDDEN_SIZE = 512
    NUM_HEADS = 16

