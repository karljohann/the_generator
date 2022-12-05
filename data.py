import glob
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

'''
# USAGE:
data = Data("/path/to/csv/") # optional parameter train_test_split=0.2
X, X_test, y, y_test = data.getTensors()
'''

class Data:
    def __init__(self, path=None, train_test_split=0.2):
        self.path = path
        self.train_test_split = train_test_split
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.df = None

        if path:
            self.parse_files()

    def parse_files(self):
        files = glob.glob(os.path.join(self.path, "*.csv"))
        data = []

        for f in files:
            df = pd.read_csv(f, index_col=0)
            data.append(df)

        self.df = pd.concat(data, axis=0, ignore_index=True)

    def getData(self, train_test_split=0.2):
        if train_test_split != 0.2:
            self.train_test_split = train_test_split # is this necessary

        mm = MinMaxScaler()
        ss = StandardScaler()

        X = self.df.drop(columns=[
            'note_int',
            'time',
            'end',
            'velocity',
        ])
        y = self.df.iloc[:, 5:6]

        X_ss = ss.fit_transform(X)
        y_mm = mm.fit_transform(y)

        split = int(X.shape[0] * (1 - self.train_test_split))

        X_train = X_ss[:split, :]
        X_test = X_ss[split:-1, :]

        y_train = y_mm[1:split+1, :]
        y_test = y_mm[split+1:, :]

        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test

        return X_train, y_train, X_test, y_test


    def getTensors(self):
        if self.X is None:
            self.getData() # otherwise there is no data to work with

        X_train_tensors = torch.Tensor(self.X)
        X_test_tensors = torch.Tensor(self.X_test)

        y_train_tensors = torch.Tensor(self.y)
        y_test_tensors = torch.Tensor(self.y_test)

        # rows, note, features
        X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
        X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

        return X_train_tensors_final, X_test_tensors_final, y_train_tensors, y_test_tensors

