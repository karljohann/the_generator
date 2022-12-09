from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset 

import matplotlib.pyplot as plt

from model import GRU_Generator
from data import Data

# DATA_FILES_PATH = "/Users/karljohann/dev/HR/the_generator/data/csv/bach/"
DATA_FILES_PATH = "/Users/karljohann/Downloads/the_generator/bach.csv"
STORAGE_PATH = "/Users/karljohann/Downloads/the_generator/"
MODEL_FILE = "the_generator_9_dec_440_epoch_bach.pk"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict():
    data = Data(csvPath=DATA_FILES_PATH)
    X_train, (y_seq, y2_seq, y3_seq), X_val, (y_val, y2_val, y3_val) = data.getData()

    # hyperparameters - batch size - lstm hidden layer size
    BATCH = 4
    num_classes = 128 # num of outputs
    num_features = 3 # input size (features)
    num_units = 8 # hidden units

    print(f'num_features: {num_features}')
    print(f'num_units: {num_units}')
    print(f'num_classes: {num_classes}')

    # seeds = [ # td, len, note
    #     [[ 0.0000,  2.7780, 76.0000]],
    #     [[ 2.7780,  0.1390, 78.0000]],
    #     [[ 2.9170,  0.1390, 76.0000]],
    #     [[ 3.0560,  0.1390, 75.0000]],
    #     [[ 3.1940,  0.1390, 76.0000]],
    #     [[ 3.3330,  0.0790, 78.0000]],
    #     [[ 3.4030,  0.0790, 76.0000]],
    #     [[ 3.4720,  0.0790, 78.0000]],
    #     [[ 3.5420,  0.0790, 76.0000]],
    #     [[ 3.6110,  0.0790, 78.0000]],
    #     [[ 3.6810,  0.4860, 76.0000]],
    #     [[ 4.1670,  0.1390, 75.0000]],
    #     [[ 4.3060,  0.1390, 76.0000]],
    #     [[ 4.4440,  4.7220, 78.0000]],
    #     [[ 9.1670,  0.2780, 76.0000]],
    #     [[ 9.4440,  0.2780, 75.0000]]
    # ]

    seeds = torch.Tensor([
        [[ 0.1470,  0.1470, 53.0000]],
        [[ 0.1470,  0.0740, 60.0000]],
        [[ 0.0740,  0.0740, 58.0000]],
        [[ 0.0740,  0.0740, 60.0000]],
    ])

    model = GRU_Generator(num_features=num_features, num_units=num_units, num_classes=num_classes)
    model.load_state_dict(torch.load(STORAGE_PATH + MODEL_FILE))

    hidden_state = model.init_states(BATCH)
    model.eval()
    
    def _predict(seeds, hidden_state):
        return model.forward(seeds, hidden_state=hidden_state)

    for _ in range(500):
        note, timedelta, note_length, hidden_state = _predict(seeds, hidden_state)

        # note_list = torch.argmax(note, dim=1).tolist()


        def get_val(t):
            return torch.argmax(t[0]).item()

        new_seed = torch.Tensor([[[
            get_val(timedelta),
            get_val(note_length),
            # max(set(note_list), key=note_list.count) # most common note
            get_val(note),
        ]]])


        seeds = torch.cat((seeds[1:(len(seeds))], new_seed))


if __name__ == "__main__":
    predict()    