import glob
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset 
import matplotlib.pyplot as plt

from model import LSTM_Generator
from data import Data

DATA_FILES_PATH = "CSV/"

def train():
    
    data = Data(path = DATA_FILES_PATH)

    X_temp, y_temp, _, _ = data.getData()

    num_classes = int(data.num_classes)
    num_features = X_temp.shape[1]

    # hyperparameters - batch size - lstm hidden layer size
    BATCH = 8
    num_units = 8

    index = np.floor(0.1*X_temp.shape[0])
    index = int(index)
    print(index)

    X_train = X_temp[index:, :]
    y_train = y_temp[index:, :]
    X_val = X_temp[:index, :]
    y_val = y_temp[:index, :]

    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    y_train = y_train.type(torch.LongTensor)
    X_val = torch.Tensor(X_val)
    y_val = torch.Tensor(y_val)
    y_val = y_val.type(torch.LongTensor)
    #reshape to rows and features
    # X_train = torch.reshape(BATCH, (1, X_train.shape[1]))

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=False, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, drop_last=True)

    print(f'num_features: {num_features}')
    print(f'num_units: {num_units}')
    print(f'num_classes: {num_classes}')

    model = LSTM_Generator(num_features=num_features, num_units=num_units, num_classes=num_classes)
    
    learning_rate = 1e-4
    num_epochs = 5

    criterion = torch.nn.CrossEntropyLoss()    # cross-entropy for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    total_train_loss = []
    total_val_loss = []
    total_epochs = []
    print('----------')
    print('Starting to train!')
    print('----------')

    for epoch in range(num_epochs):
        
        total_epochs.append(epoch)
        print(f'epoch: {epoch+1}')

        hidden_state, cell_state = model.init_states(BATCH)

        train_loss = 0
        train_correct = 0
        train_total = 0

        batch_iter = 1
        model.train()
        for train_data in train_dl:
            batch_iter += 1
            X_train, y_train = train_data
            X_train = X_train.reshape(-1, 1, X_train.shape[1])
            y_train = torch.squeeze(y_train)

            optimizer.zero_grad() #caluclate the gradient, manually setting to 0

            output, (hidden_state, cell_state) = model.forward(X_train, prev_state=(hidden_state, cell_state))    #.view(-1, X_train.shape[1])) #forward pass
            

            for index, value in enumerate(output):
                if torch.argmax(value) == y_train[index]:
                    train_correct += 1
                train_total += 1

            # obtain the loss function
            loss = criterion(output, y_train)
            loss_value = loss.item()
            train_loss += loss_value
            # perform backpropagation
            loss.backward(retain_graph=True) #calculates the loss of the loss function
            # print(hidden_state.shape)
            hidden_state, cell_state = hidden_state.detach(), cell_state.detach()
 
            optimizer.step() #improve from loss, i.e backprop
            
            if batch_iter % 1000 == 0:
                print('Epoch: {}/{}'.format(epoch+1, num_epochs),
                          'Iteration: {}'.format(batch_iter),
                          'Training Loss: {}'.format(loss_value),
                          'Training Accuracy: {}'.format(round(train_correct/train_total, 3)))
        

        print('\n Validating! \n')
        hidden_state, cell_state = model.init_states(BATCH)

        val_loss = 0
        val_correct = 0
        val_total = 0

        batch_iter = 1
        model.eval()
        with torch.no_grad(): 
            for train_data in train_dl:
                batch_iter += 1
                X_val, y_val = train_data
                X_val = X_val.reshape(-1, 1, X_val.shape[1])
                y_val = torch.squeeze(y_val)

                output, (hidden_state, cell_state) = model.forward(X_val, prev_state=(hidden_state, cell_state))    #.view(-1, X_train.shape[1])) #forward pass

                for index, value in enumerate(output):
                    if torch.argmax(value) == y_val[index]:
                        val_correct += 1
                    val_total += 1

                # obtain the loss function
                loss = criterion(output, y_val)
                loss_value = loss.item()
                val_loss += loss_value
    
                if batch_iter % 1000 == 0:
                    print('Epoch: {}/{}'.format(epoch+1, num_epochs),
                            'Iteration: {}'.format(batch_iter),
                            'Validation Loss: {}'.format(loss_value),
                            'Validation Accuracy: {}'.format(round(train_correct/train_total, 3)))              
            # if epoch % 100 == 0:
        # save loss and accuracy for this epoch
        total_train_loss.append(train_loss/len(train_dl))
        print("Epoch: %d, training loss: %1.5f" % (epoch+1, train_loss/len(train_dl))) 
        print(f'training acc: {round(train_correct/train_total, 3)}')
        total_val_loss.append(val_loss/len(val_dl))
        print("Epoch: %d, validation loss: %1.5f" % (epoch+1, val_loss/len(val_dl))) 
        print(f'validation acc: {round(val_correct/val_total, 3)}')

    plt.plot(total_epochs, total_train_loss)
    plt.plot(total_epochs, total_val_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training  and validationLoss')
    plt.show()


# class timeseries_dataset(Dataset):
#    def __init__(self,X,y):
#        self.x = torch.tensor(X,dtype=torch.float32)
#        self.y = torch.tensor(y,dtype=torch.float32)
#        self.len = X.shape[0]
#
#    def __getitem__(self,idx):
#        return self.X[idx],self.y[idx]
#  
#    def __len__(self):
#        return self.len        

if __name__ == "__main__":
    train()    