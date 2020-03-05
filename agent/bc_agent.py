import torch
import torch.nn as nn
import torch.optim as optim
from agent.networks import CNN
from agent.networks import FCN
from agent.networks import ResNet
from agent.networks import LSTM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BCAgent:

    def __init__(self, network_type, lr, hidden_layers, history_length):
        # TODO: Define network, loss function, optimizer
        # self.net = FCN(...) or CNN(...)
        self.network_type = network_type
        self.lr = lr
        self.hidden_layers = hidden_layers
        self.history_length = history_length
        self.layer_dim = 2
        self.output_dim = 4
        self.seq_dim = 100
        self.input_dim = 100
        self.hidden_dim = 100

        if self.network_type == "FCN":
            self.net = FCN(self.hidden_layers).to(device)
        elif self.network_type == "CCN":
            self.net = CNN(self.history_length).to(device)
        elif self.network_type == 'LSTM':
            self.net = LSTM(self.input_dim, self.hidden_dim, self.layer_dim, self.output_dim)
        else:
            self.net = ResNet(self.history_length, num_layers=[2,2,2,2]).to(device)

        self.loss_fcn = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.net.parameters(), lr)

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        # TODO: forward + backward + optimize
        X_batch = torch.tensor(X_batch).to(device)
        y_batch = torch.FloatTensor(y_batch).to(device)
        X_batch = X_batch.view(-1, self.input_dim, self.seq_dim).requires_grad_()
        self.net.zero_grad()
        output = self.net(X_batch)
        y_batch = y_batch.view(y_batch.size(0))
        loss = self.loss_fcn(output, y_batch.long())
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self, X):
        # TODO: forward pass
        X = X.view(-1, self.input_dim, self.seq_dim).to(device)
        outputs = self.net(X)
        return outputs

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name, map_location=device))
