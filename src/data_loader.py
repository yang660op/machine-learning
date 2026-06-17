import pandas as pd
from src.config import TRAIN_FILE, TEST_FILE, VAL_FILE


def load_data():
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    val = pd.read_csv(VAL_FILE)
    return train, test, val


def load_train():
    return pd.read_csv(TRAIN_FILE)


def load_test():
    return pd.read_csv(TEST_FILE)


def load_val():
    return pd.read_csv(VAL_FILE)
