import glob
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset 
import matplotlib.pyplot as plt

from model import LSTM_Generator
from data import process_data

DATA_FILES_PATH = "CSV/"

def train():
    files = glob.glob(os.path.join(DATA_FILES_PATH, "*.csv"))
    data = []

    for f in files:
        df = pd.read_csv(f, index_col=0)
        data.append(df)

    X_train, y_train = process_data(data)

    num_classes = y_train.nunique()
    num_features = X_train.shape[1]

    # print(X_train[:5])

    X_train = torch.Tensor(X_train.values)
    y_train = torch.Tensor(y_train.values)
    y_train = y_train.type(torch.LongTensor)
    #reshape to rows and features
    # X_train = torch.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    train_ds = TensorDataset(X_train, y_train)
    BATCH = 32
    num_units = 16

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=False)

    model = LSTM_Generator(num_features=num_features, num_units=num_units, num_classes=num_classes)
    
    learning_rate = 1e-3
    num_epochs = 10

    criterion = torch.nn.CrossEntropyLoss()    # cross-entropy for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    total_train_loss = []
    total_epochs = []
    print('----------')
    print('Starting to train!')
    print('----------')

    for epoch in range(num_epochs):
        train_loss = 0
        total_epochs.append(epoch)
        model.train()
        for train_data in train_dl:
            X_train, y_train = train_data
            # test = X_train.view(-1, X_train.shape[1])
            # print(X_train.shape)
            # print(test.shape)
            # print('----------')
            # print(X_train.view(-1, X_train.shape[1]))
            # print('----------')
            X_train = torch.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))     
            outputs = model.forward(X_train)    #.view(-1, X_train.shape[1])) #forward pass
            optimizer.zero_grad() #caluclate the gradient, manually setting to 0

            # obtain the loss function
            loss = criterion(outputs, y_train)
            loss.backward() #calculates the loss of the loss function
 
            optimizer.step() #improve from loss, i.e backprop
            train_loss += loss.item()
            # if epoch % 100 == 0:
        # save loss and accuracy for this epoch
        total_train_loss.append(train_loss/len(train_dl))
        print("Epoch: %d, loss: %1.5f" % (epoch, train_loss/len(train_dl))) 

    plt.plot(total_epochs, total_train_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


class timeseries_dataset(Dataset):
    def __init__(self,X,y):
        self.x = torch.tensor(X,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.len = X.shape[0]

    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]
  
    def __len__(self):
        return self.len        

if __name__ == "__main__":
    train()    