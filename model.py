import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRU_Generator(nn.Module):
    def __init__(self, num_features, num_units, num_classes, num_layers=1):
        super(GRU_Generator, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of GRU layers
        self.input_size = num_features #input size (features)
        self.hidden_size = num_units #hidden units
        # self.seq_length = seq_length #sequence length

        self.gru_layer = nn.GRU(
                                input_size=self.input_size, 
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers, 
                                batch_first=True
                          )
        self.dense_layer1 =  nn.Linear(in_features=self.hidden_size, out_features=128) #fully connected 1
        self.dense_layer2 =  nn.Linear(in_features=self.hidden_size, out_features=128) #fully connected 1
        self.dense_layer3 =  nn.Linear(in_features=self.hidden_size, out_features=128) #fully connected 1
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.output_layer1 = nn.Linear(in_features=self.hidden_size, out_features=num_classes) #fully connected last layer
        # self.output_layer2 = nn.Linear(in_features=self.hidden_size, out_features=1) #fully connected last layer
        self.output_layer3 = nn.Linear(in_features=self.hidden_size, out_features=1) #fully connected last layer
        self.output_layer2 = nn.Sigmoid()
        
    
    def forward(self,x, hidden_state):
        # Propagate input through GRU
        out, new_state = self.gru_layer(x, hidden_state) #lstm with input, hidden, and internal state
        # out = out.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = out[:, -1, :]
        out1 = self.dense_layer1(out) #first Dense
        out2 = self.dense_layer2(out) #first Dense
        out3 = self.dense_layer3(out) #first Dense
        out1 = self.relu1(out1) #relu
        out2 = self.relu2(out2) #relu
        out3 = self.relu3(out3) #relu
        out1 = self.output_layer1(out) #Final Output
        out2 = self.output_layer2(out) #Final Output
        out3 = self.output_layer3(out) #Final Output

        return out1, out2, out3, new_state

    def init_states(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device) #hidden state
        # c0 = torch.zeros(self.num_layers, num_samples, self.hidden_size) ##internal cell state

        return h0
