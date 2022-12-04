import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# TODO: Move STORAGE_PATH to utils.py file

DATA_FILES_PATH = "/Users/karljohann/Downloads/the_generator/csv/"
STORAGE_PATH = "/Users/karljohann/Downloads/the_generator/"


def process_data(data):
    df = pd.concat(data, axis=0, ignore_index=True)
    train_set, test_set = train_test_split(df, test_size=0.1, random_state=123)

    # train_set.info()
    train_set.summary()

    y = train_set['note_int'].copy()
    X = train_set.drop(columns=[
        'note_int',
        'time',
        'end',
        'velocity',
    ])
    # X = train_set.drop('note_int', axis=1)

    return X, y



if __name__ == "__main__":
    files = glob.glob(os.path.join(DATA_FILES_PATH, "*.csv"))
    data = []

    for f in files:
        df = pd.read_csv(f, index_col=0)
        data.append(df)

    process_data(data)
