from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset 

import matplotlib.pyplot as plt
import pretty_midi as pm

from model import GRU_Generator
from data import Data

DATA_FILES_PATH = ""
STORAGE_PATH = ""
MODEL_FILE = ""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict():
    data = Data(csvPath=DATA_FILES_PATH)
    X_pred, _, _, _ = data.getData()

    num_notes = 440

    # hyperparameters - batch size - lstm hidden layer size
    BATCH = 16
    num_classes = 128 # num of outputs
    num_features = 3 # input size (features)
    num_units = 8 # hidden units

    pred_dl = DataLoader(X_pred, batch_size=BATCH, shuffle=False, drop_last=True)

    model = GRU_Generator(num_features=num_features, num_units=num_units, num_classes=num_classes)
    model.load_state_dict(torch.load(STORAGE_PATH + MODEL_FILE))

    hidden_state = model.init_states(BATCH)
    model.eval()

    def get_val(t):
        return torch.argmax(t[0]).item()

    # Create a PrettyMIDI object
    tester = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program('Acoustic Grand Piano')
    piano = pm.Instrument(program=piano_program)
    start_time = 0.0

    n = 0
    with torch.no_grad():
        for pred_data in pred_dl:
            X_val = pred_data
            X_val = X_val.reshape(-1, 1, X_val.shape[1])
            # print(X_val)

            note, timedelta, note_length, hidden_state = model.forward(X_val, hidden_state=hidden_state)

            _timedelta = float(get_val(timedelta) / 4)
            start_time += _timedelta
            _note_length = get_val(note_length)
            new_note = pm.Note(velocity=100, pitch=get_val(note), start=start_time, end=start_time+_note_length)
            piano.notes.append(new_note)

            if n >= num_notes:
                break
            n += 1

    tester.instruments.append(piano)
    tester.write(STORAGE_PATH + f"bach_440_{datetime.timestamp(datetime.now())}.mid") # write out the MIDI data


if __name__ == "__main__":
    predict()
