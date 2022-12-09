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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    data = Data(csvPath=DATA_FILES_PATH)
    # data = Data(path=DATA_FILES_PATH + "small_sax.pkl", isPkl=True)
    X_train, (y_seq, y2_seq, y3_seq), X_val, (y_val, y2_val, y3_val) = data.getData()
    # X_seq, y_seq, y2_seq, y3_seq = data.getSequence(seq_len=4)
    X_seq = X_train


    # hyperparameters - batch size - lstm hidden layer size
    BATCH = 16
    num_classes = data.num_classes # num of outputs
    num_features = X_seq.shape[1] # input size (features)
    num_units = 8 # hidden units

    train_ds = TensorDataset(X_seq, y_seq, y2_seq, y3_seq)
    test_ds = TensorDataset(X_val, y_val, y2_val, y3_val)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=False, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False, drop_last=True)

    print(f'num_features: {num_features}')
    print(f'num_units: {num_units}')
    print(f'num_classes: {num_classes}')

    model = GRU_Generator(num_features=num_features, num_units=num_units, num_classes=num_classes)
    
    num_epochs = 440

    # learning_rate = 1e-3
    learning_rate = 0.002
    # lambda1 = lambda epoch: epoch // 30
    # lambda2 = lambda epoch: 0.95 ** epoch

    criterion = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.MSELoss()
    criterion3 = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)

    note_train_acc = []
    note_test_acc = []
    total_train_loss = []
    total_train2_loss = []
    total_train3_loss = []
    total_val_loss = []
    total_epochs = []
    print("----------", "\nStarting to train", "\n----------")

    def getLoss(_output, y, _criterion, train=True):
        correct = 0
        train_total = 0

        for index, value in enumerate(_output):
            # print(torch.argmax(value).item(), y_sq.item())
            # return
            # print(">>>", torch.argmax(value), torch.argmax(y_sq[index]))
            # if index % 100000 == 0:
            #     print(">>> value, y_sq", torch.argmax(value), y[index])
            # if torch.argmax(value) == torch.argmax(y_sq[index]):
            if torch.argmax(value).item() == y[index].item():
                correct += 1

            # if show:
            #     print(
            #         torch.argmax(value).item(),
            #         "=",
            #         y[index].item(),
            #         end=", "
            #     )
            train_total += 1

        loss = _criterion(_output, y)

        if train:
            loss.backward(retain_graph=True) #calculates the loss of the loss function
    
        return loss.item(), train_total, correct

    for epoch in range(num_epochs):
        
        total_epochs.append(epoch)

        hidden_state = model.init_states(BATCH)

        train_loss = 0
        train2_loss = 0
        train3_loss = 0
        train_correct = 0
        train_total = 0

        batch_iter = 1
        model.train()

        for train_data in train_dl:
            batch_iter += 1
            X_train, y_seq, y2_seq, y3_seq = train_data
            X_sq = X_train.reshape(-1, 1, X_train.shape[1])

            optimizer.zero_grad()

            if (epoch + 1) == num_epochs:
                print(X_sq)

            output1, output2, output3, hidden_state = model.forward(X_sq, hidden_state=hidden_state)

            loss_value, _train_total, _train_correct = getLoss(output1, y_seq, criterion)
            loss2_value, _, _ = getLoss(output2, y2_seq, criterion2)
            loss3_value, _, _ = getLoss(output3, y3_seq, criterion3)

            train_total += _train_total
            train_correct += _train_correct
            train_loss += loss_value
            train2_loss += loss2_value
            train3_loss += loss3_value

            # grad clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            hidden_state = hidden_state.detach()
            optimizer.step()

        # scheduler.step()


        # save loss and accuracy for this epoch
        note_train_acc.append(train_correct/train_total)
        total_train_loss.append(train_loss/len(train_dl))
        total_train2_loss.append(train2_loss/len(train_dl))
        total_train3_loss.append(train3_loss/len(train_dl))

        print(f"Epoch: {epoch+1}/{num_epochs}",
            f"Loss: |{round(train_loss, 3)}| (note), |{round(train2_loss, 3)}| (timedelta), |{round(train3_loss, 3)}| (note_length)\t",
            f"Acc: {round(train_correct/train_total, 3)} (note)"
        )


        ## VALIDATION
        # if (epoch + 1) == num_epochs:
        # print('\n Validating! \n')
        hidden_state = model.init_states(BATCH)

        val_loss = 0
        val2_loss = 0
        val3_loss = 0
        val_correct = 0
        val_total = 0

        batch_iter = 1
        model.eval()
        with torch.no_grad(): 
            for test_data in test_dl:
                batch_iter += 1
                X_val, y_val, y2_val, y3_val = test_data
                X_val = X_val.reshape(-1, 1, X_val.shape[1])
                y_val = torch.squeeze(y_val)

                output1, output2, output3, hidden_state = model.forward(X_val, hidden_state=hidden_state)

                loss_value, _test_total, _test_correct = getLoss(output1, y_val, criterion, False)
                loss_value2, _, _ = getLoss(output2, y2_val, criterion2, False)
                loss_value3, _, _ = getLoss(output3, y3_val, criterion3, False)

                val_correct += _test_correct
                val_total += _test_total

                val_loss += loss_value
                val2_loss += loss_value2
                val3_loss += loss_value3

        # print(f"Epoch: {epoch+1}/{num_epochs}",
        #     f"Val Loss: |{round(val_loss, 3)}| (note), |{round(val2_loss, 3)}| (timedelta), |{round(val3_loss, 3)}| (note_length)\t",
        #     f"Val Acc: {round(val_correct/val_total, 3)} (note)"
        # )

        note_test_acc.append(val_correct/val_total)
        total_val_loss.append(val_loss/len(test_dl))
        # if epoch % 100 == 0:
        # print(f"Validation loss: {val_loss/len(test_dl)}, {val2_loss/len(test_dl)}, {val3_loss/len(test_dl)}") 
        # print(f'validation acc: {round(val_correct/val_total, 3)}')


    ## SAVE MODEL
    torch.save(model.state_dict(), STORAGE_PATH + f"the_generator_{datetime.timestamp(datetime.now())}.pk")

    ## PLOT SHIT UP
    fig, axs = plt.subplots(4, sharex=True, sharey=False)
    axs[0].set_title('Note training accuracy')
    axs[0].plot(total_epochs, note_train_acc)
    axs[0].plot(total_epochs, note_test_acc)
    axs[1].set_title('Note training loss')
    axs[1].plot(total_epochs, total_train_loss)
    axs[1].plot(total_epochs, total_val_loss)
    axs[2].set_title('Timedelta training loss')
    axs[2].plot(total_epochs, total_train2_loss)
    axs[3].set_title('Note length training loss')
    axs[3].plot(total_epochs, total_train3_loss)
    # plt.plot(total_epochs, total_val_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.show()


if __name__ == "__main__":
    train()    