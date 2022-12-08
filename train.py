import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset 

import matplotlib.pyplot as plt

from model import GRU_Generator
from data import Data

# DATA_FILES_PATH = "/Users/karljohann/dev/HR/the_generator/data/csv/bach/"
DATA_FILES_PATH = "/Users/karljohann/Downloads/the_generator/bach.csv"
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
    num_units = 16 # hidden units

    train_ds = TensorDataset(X_seq, y_seq, y2_seq, y3_seq)
    test_ds = TensorDataset(X_val, y_val, y2_val, y3_val)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=False, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False, drop_last=True)

    print(f'num_features: {num_features}')
    print(f'num_units: {num_units}')
    print(f'num_classes: {num_classes}')

    model = GRU_Generator(num_features=num_features, num_units=num_units, num_classes=num_classes)
    
    num_epochs = 1200

    learning_rate = 1e-4
    lr_lambda = 0.99

    criterion = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.MSELoss()
    criterion3 = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: lr_lambda ** epoch)

    total_train_loss = []
    total_train2_loss = []
    total_train3_loss = []
    total_val_loss = []
    total_epochs = []
    print("----------", "\nStarting to train", "\n----------")

    def grad_clipping(net, theta):  
        ''' Clip the gradient '''
        params = [p for p in net.parameters() if p.requires_grad]

        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm

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
            if torch.argmax(value) == y[index]:
            # if torch.argmax(value).item() == y_sq.item():
                correct += 1
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

            # print(">>>>> input size", y2_seq.unsqueeze(1).shape, y3_seq.unsqueeze(1).shape)
            # return

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

        scheduler.step()

        print(f"Epoch: {epoch+1}/{num_epochs}",
            f"Loss: {round(train_loss, 3)} |{round(train_loss/len(train_dl), 3)}| (note), {round(train2_loss, 3)} |{round(train2_loss/len(train_dl), 3)}| (timedelta), {round(train3_loss, 3)} |{round(train3_loss/len(train_dl), 3)}| (note_length)\t",
            f"Acc: {round(train_correct/train_total, 3)} (note)"
        )


        ## VALIDATION
        if (epoch + 1) == num_epochs:
            print('\n Validating! \n')
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
        
                    if batch_iter % 1000 == 0:
                        print('Epoch: {}/{}'.format(epoch+1, num_epochs),
                                'Iteration: {}'.format(batch_iter),
                                'Validation Loss: {}'.format(loss_value),
                                'Validation Accuracy: {}'.format(round(train_correct/train_total, 3)))

                # if epoch % 100 == 0:
            total_val_loss.append(val_loss/len(test_dl))
            print(f"Validation loss: {val_loss/len(test_dl)}, {val2_loss/len(test_dl)}, {val3_loss/len(test_dl)}") 
            # print(f'validation acc: {round(val_correct/val_total, 3)}')


        # save loss and accuracy for this epoch
        total_train_loss.append(train_loss/len(train_dl))
        total_train2_loss.append(train2_loss/len(train_dl))
        total_train3_loss.append(train3_loss/len(train_dl))
        # print("Epoch: %d, training loss: %1.5f" % (epoch+1, train_loss/len(train_dl))) 
        # print(f'training acc: {round(train_correct/train_total, 3)}')

    fig, axs = plt.subplots(3, sharex=True, sharey=False)
    axs[0].plot(total_epochs, total_train_loss)
    axs[0].set_title('Note training loss')
    axs[1].plot(total_epochs, total_train2_loss)
    axs[1].set_title('Timedelta training loss')
    axs[2].plot(total_epochs, total_train3_loss)
    axs[2].set_title('Note length training loss')
    # plt.plot(total_epochs, total_val_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.show()


if __name__ == "__main__":
    train()    