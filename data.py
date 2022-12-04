import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# TODO: Move STORAGE_PATH, output_file, and get_files to utils.py file

DATA_FILES_PATH = "/Users/karljohann/Downloads/the_generator/csv/"
STORAGE_PATH = "/Users/karljohann/Downloads/the_generator/"

def get_files(filetype, dir = None):
    files = []
    for file_path in list(glob.glob(dir)):
        _, ext = os.path.splitext(file_path)
        if ext == '.mid':
            files.append(file_path)
        if os.path.isdir(file_path):
            files += get_files(filetype, file_path + f"/*.{filetype}")

    return files

def output_file(filename, filetype, appendix=""):
    return f"{STORAGE_PATH}{Path(filename).stem}{appendix}.{filetype}"



def process_data(df):
    train_set, test_set = train_test_split(df, test_size=0.1, random_state=123)

    y = train_set['note_int'].copy()
    X = train_set.drop('note_int')



if __name__ == "__main__":
    files = get_files("csv", DATA_FILES_PATH + "/**") # all files and subfolders
    data = []

    for i, f in enumerate(files):
        df = pd.read_csv(f, index_col=0)
        data.append(df)
    
    process_data(data)
