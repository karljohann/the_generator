import glob
import os
import pickle

import pandas as pd
import numpy as np
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

'''
# USAGE:
data = Data("/path/to/csv/") # optional parameter train_test_split=0.2
X, X_test, y, y_test = data.getTensors()
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Data:
    def __init__(self,
            path=None,
            train_test_split=0.2,
            isPkl=False,
            pklPath=None,
            save_to_csv=False,
            isSequence=False,
            csvPath=None,
    ):
        self.path = path
        self.train_test_split = train_test_split
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.df = None
        self.num_classes = None
        self.isPkl = isPkl
        self.pklPath = pklPath
        self.save_to_csv = save_to_csv
        self.isSequence = isSequence

        if csvPath:
          self.df = pd.read_csv(csvPath)
        elif path:
            self.parseFiles()

    def parseFiles(self):
        if self.isPkl:
            self.df = pd.read_pickle(self.path)
        else:
            files = glob.glob(os.path.join(self.path, "*.csv"))
            data = []

            for f in files:
                df = pd.read_csv(f)
                data.append(df)

            self.df = pd.concat(data, axis=0, ignore_index=True)

            if self.save_to_csv:
                self.df.to_csv("", index=False)

            if self.pklPath is not None:
                f = open(self.pklPath, 'wb')
                pickle.dump(self.df, f)
                f.close()

    def getData(self, train_test_split=0.2):
        if train_test_split != self.train_test_split:
            self.train_test_split = train_test_split

        X = self.df.drop(columns=[
            # 'note_int',
            'instrument',
            'instrument_name',
            'note',
            'time',
            'end',
            'velocity',
            # 'timedelta',
            # 'note_length',
            'tick',
            'tickdelta',
            'note_length_tick',
        ])

        X_ss = X.values

        split = int(X.shape[0] * (1 - self.train_test_split))

        X_train = X_ss[:split, :]
        X_test = X_ss[split:-1, :]

        def get_y(tp):
            cols = ['note_int', 'timedelta', 'note_length']
            y_ret = []
            for col in cols:
                y = self.df.loc[:, col]
                if tp == 'train':
                    y_ret.append(y.values[1:split+1])
                else:
                    y_ret.append(y.values[split+1:])
                    

            return (y_ret[0], y_ret[1], y_ret[2])

        y_train, y2_train, y3_train = get_y('train')
        y_test, y2_test, y3_test = get_y('test')

        # self.num_classes = (self.df.loc[:, 'note_int'].nunique() + 10)
        self.num_classes = 128

        self.X = X_train
        self.y = (y_train, y2_train, y3_train)
        self.X_test = X_test
        self.y_test = (y_test, y2_test, y3_test)

        return (
            torch.tensor(X_train, dtype=torch.float32).squeeze().to(device),
            (
                torch.tensor(y_train, dtype=torch.long).squeeze().to(device),
                torch.tensor(y2_train, dtype=torch.float32).unsqueeze(1).to(device),
                torch.tensor(y3_train, dtype=torch.float32).unsqueeze(1).to(device),
            ),
            torch.tensor(X_test, dtype=torch.float32).squeeze().to(device),
            (
                torch.tensor(y_test, dtype=torch.long).squeeze().to(device),
                torch.tensor(y2_test, dtype=torch.float32).unsqueeze(1).to(device),
                torch.tensor(y3_test, dtype=torch.float32).unsqueeze(1).to(device),
            ),
        )


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

    def inverseTransform(self, y_pred, y):
        return self.mm.inverse_transform(y_pred), self.mm.inverse_transform(y)
    
    def getSequence(self, seq_len = 4):
        # FIXME: should do for each song at a time, otherwise the last note of one song is predicting the first of the next
        if self.X is None:
            self.getData() # otherwise there is no data to work with ... FIXME: Should really just figure this out

        X_seq = []
        y_seq = []
        y2_seq = []
        y3_seq = []
        for i in range(seq_len, len(self.X)):
            if i + (seq_len * 2) >= len(self.X):
                break

            X_seq_tmp = []
            y_seq_tmp = []
            y2_seq_tmp = []
            y3_seq_tmp = []
            for j in range(1, seq_len+1):
                X_seq_tmp.append(self.X[i + j:i + j + seq_len, :])
                y_seq_tmp.append(self.y[i + j + seq_len, :])
                y2_seq_tmp.append(self.X[i + j + seq_len, 0:1])
                y3_seq_tmp.append(self.X[i + j + seq_len, 1:2])

            X_seq.append(np.array(X_seq_tmp))
            y_seq.append(y_seq_tmp)
            y2_seq.append(y2_seq_tmp)
            y3_seq.append(y3_seq_tmp)
        
        return (
            torch.tensor(X_seq, dtype=torch.float32).squeeze().to(device),
            torch.tensor(y_seq, dtype=torch.long).squeeze().to(device),
            torch.tensor(y2_seq, dtype=torch.float32).squeeze().to(device),
            torch.tensor(y3_seq, dtype=torch.float32).squeeze().to(device),
        )
        
