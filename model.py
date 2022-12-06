import torch
import torch.nn as nn

'''
class LSTM_Generator(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_classes=1):
        super(LSTM_Generator, self).__init__()
        self.num_classes = num_classes #number of classes (unique notes)
        self.input_size = input_size #input size (number of features)
        self.hidden_size = hidden_size #hidden states

        self.lstm_layer1 = nn.LSTM(input_size, hidden_size, batch_first = True)
        # self.lstm_layer2 = nn.LSTMCell(hidden_size, hidden_size)                  
        self.relu1 = nn.ReLU()

        self.dense_layer =  nn.Linear(hidden_size, 128) #fully connected hidden layer
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(128, num_classes) #fully connected output layer
        
    def forward(self, x):
        outputs = []
        print('---x----')
        print(x)
        print('-------')
        num_samples = x.size(0)

        hidden_state1 = torch.zeros(num_samples, self.hidden_size) # hidden memory state
        cell_state1 = torch.zeros(num_samples, self.hidden_size) # internal cell state
        hidden_state2 = torch.zeros(num_samples, self.hidden_size) # hidden memory state
        cell_state2 = torch.zeros(num_samples, self.hidden_size) # internal cell state
        # Propagate input through the LSTM network
        hidden_state1, cell_state1 = self.lstm_layer1(x, (hidden_state1, cell_state1)) #lstm with input, hidden, and internal state
        # hidden_state2, cell_state2 = self.lstm_layer2(hidden_state1, (hidden_state2, cell_state2))
        hidden_state1 = hidden_state1.view(-1, self.hidden_size) #reshape the data for the dense layer

        out = self.relu1(hidden_state1)
        out = self.dense_layer(out) 
        out = self.relu2(out) 
        out = self.output_layer(out) 
        return out    
'''

class LSTM_Generator(nn.Module):
    def __init__(self, num_features, num_units, num_classes, num_layers=1):
        super(LSTM_Generator, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of lstm layers
        self.input_size = num_features #input size (features)
        self.hidden_size = num_units #hidden units
        # self.seq_length = seq_length #sequence length

        self.lstm_layer = nn.LSTM(
                                input_size=self.input_size, 
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers, 
                                batch_first=True
                          ) #lstm
        self.dense_layer =  nn.Linear(in_features=self.hidden_size, out_features=128) #fully connected 1
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(in_features=128, out_features=num_classes) #fully connected last layer
        
    
    def forward(self,x, prev_state):
        # initialize states
        hidden_state, cell_state = prev_state
        # Propagate input through LSTM
        output, new_state = self.lstm_layer(x, (hidden_state, cell_state)) #lstm with input, hidden, and internal state

        out = output.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(out)
        out = self.dense_layer(out) #first Dense
        out = self.relu(out) #relu
        out = self.output_layer(out) #Final Output

        return out, new_state    

    def init_states(self, batch_size):
        # num_samples = x.size(0)
        num_samples = batch_size
        h0 = torch.zeros(self.num_layers, num_samples, self.hidden_size) #hidden state
        c0 = torch.zeros(self.num_layers, num_samples, self.hidden_size) ##internal cell state
        # print('h0:')
        # print(h0)
        # print('c0:')
        # print(c0)

        return h0, c0