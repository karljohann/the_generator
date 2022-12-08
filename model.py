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
        self.dense_layer =  nn.Linear(in_features=self.hidden_size, out_features=128) #fully connected 1
        self.relu = nn.ReLU()
        self.output_layer1 = nn.Linear(in_features=128, out_features=num_classes) #fully connected last layer
        self.output_layer2a = nn.Linear(in_features=128, out_features=1) #fully connected last layer
        self.output_layer2 = nn.Sigmoid()
        
    
    def forward(self,x, hidden_state):
        # Propagate input through GRU
        out, new_state = self.gru_layer(x, hidden_state) #lstm with input, hidden, and internal state
        # out = out.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = out[:, -1, :]
        out = self.dense_layer(out) #first Dense
        out = self.relu(out) #relu
        out1 = self.output_layer1(out) #Final Output
        out2 = self.output_layer2a(out) #Final Output
        out2 = self.output_layer2(out2) #Final Output
        out3 = self.output_layer2a(out) #Final Output
        out3 = self.output_layer2(out2) #Final Output

        return out1, out2, out3, new_state

    def init_states(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device) #hidden state
        # c0 = torch.zeros(self.num_layers, num_samples, self.hidden_size) ##internal cell state

        return h0
