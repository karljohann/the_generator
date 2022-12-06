from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

from data import Data
from model import LSTM1


STORAGE_PATH = "/Users/karljohann/Downloads/the_generator/csv/"

if __name__ == "__main__":
    data = Data(STORAGE_PATH, train_test_split=0.4)
    X, X_test, y, y_test = data.getTensors()


    ## TRAIN
    num_epochs = 20
    learning_rate = 0.0001

    input_size = 5 # number of features
    hidden_size = 2 # number of features in hidden state
    num_layers = 1 # number of stacked lstm layers

    num_classes = 1 # number of output classes
    # num_classes = 128 # number of output classes

    lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X.shape[1]) # our lstm class

    # criterion = torch.nn.MSELoss()    # mean-squared error for regression
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)


    for epoch in range(num_epochs):
        outputs = lstm1.forward(X) # forward pass
        optimizer.zero_grad() # calculate the gradient, manually setting to 0
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step() # improve from loss, i.e backprop
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


    ## SAVE MODEL
    # torch.save(lstm1.state_dict(), STORAGE_PATH + "lstm1_5dec_crossentropy_250epochs.pt")


    ## PREDICT


    # USEFUL
    '''
    train_predict = lstm1(X_test_tensors_final)#forward pass
    data_predict = train_predict.data.numpy() #numpy conversion
    dataY_plot = y_test_tensors.data.numpy()

    data_predict = mm.inverse_transform(data_predict) #reverse transformation
    dataY_plot = mm.inverse_transform(dataY_plot)
    plt.figure(figsize=(10,6)) #plotting
    plt.axvline(x=200, c='r', linestyle='--') #size of the training set

    plt.plot(dataY_plot, label='Actuall Data') #actual plot
    plt.plot(data_predict, label='Predicted Data') #predicted plot
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.show()
    '''


    # ACTUAL PREDICTION

    correct = 0
    with torch.no_grad():
        train_predict = lstm1(X_test)
        data_predict = train_predict.data.numpy()
        dataY_plot = y_test.data.numpy()

        # data_predict = mm.inverse_transform(data_predict)
        # dataY_plot = mm.inverse_transform(dataY_plot)
        data_predict, dataY_plot = data.inverseTransform(data_predict, dataY_plot) # reverse transformation

        for i, val in enumerate(data_predict):
            # print(i, val, data_predict[i], dataY_plot[i])
            if round(data_predict[i][0], 0) == round(dataY_plot[i][0], 0):
                correct += 1

            # if i % 10 == 0:
            #   print(data_predict[i], dataY_plot[i])

        print(f"Accuracy: {round(correct/len(train_predict), 3)}")
        # print(f"Accuracy: {round(correct/len(train_predict) * 100, 2)}%")




    '''
    total = 0
    correct = 0
    with torch.no_grad():   # No need for keepnig track of necessary changes to the gradient.
        y = y_test_tensors
        output = lstm1(X_test_tensors_final)
        # for data in X_test_tensors_final:
        #     print('>>> data',data)
        #     # X = data
        #     # print('DF_YYY', df_y_mm)
        #     output = lstm1(data)
            # print('----- OUTPUT -----', output)
        for idx, val in enumerate(output):
            # print(int(y[idx][0]), int(val[0]))
            # print('YASERASD', torch.argmax(val))
            # if torch.argmax(val) == y[idx]:
            # print('Y', val.item(), y[idx].item(), (val[0]), (y[idx][0]))
            # print(f"Comparing X with y, ({val} == {y[idx]})")
            # print(f"X == y => ({round(val.item(), 0)} == {round(y[idx].item(), 0)})")
            # print(f"{round(val.item(), 0)},")
            x_val = val.item()
            y_val = y[idx].item()
            # print(f"X == y => ({round(x_val, 0)} == {round(y_val, 0)})")
            if argmax(x_val), 0) == round(y_val, 0):
            # if round(abs(x_val), 0) == round(y_val, 0):
                # print("+correct", val, idx)
                correct += 1
            # else:
            #     print(f"X with y, {round(x_val, 3)} == {y_val})")
            total += 1
    print(f"Accuracy: {round(correct/total * 100, 2)}%")

    '''
