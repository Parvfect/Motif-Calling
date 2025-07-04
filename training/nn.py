
import torch
import torch.nn as nn
from torch.nn import Conv1d, MaxPool1d, GRU, Linear, CTCLoss
import torch.nn.functional as F


class ConvolutionalEncoder(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super(ConvolutionalEncoder, self).__init__()
        self.layer_1 = Conv1d(
            in_channels=1, out_channels=4, stride=1, kernel_size=5)
        self.layer_2 = Conv1d(
            in_channels=4, out_channels=16, stride=1, kernel_size=5)
        self.layer_3 = Conv1d(
            in_channels=16, out_channels=64, stride=2, kernel_size=5)
        self.layer_4 = Conv1d(
            in_channels=64, out_channels=hidden_size, stride=2, kernel_size=19)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        x = x.permute(2, 0, 1)
        return x
    

class RNNLayers(nn.Module):
    def __init__(self, hidden_size, num_layers, n_classes):
        super(RNNLayers, self).__init__()

        self.bigru = GRU(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True)

        self.output = Linear(hidden_size*2, n_classes)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

    
    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        x, _ = self.bigru(x, h0)
        return F.log_softmax(self.output(x), dim=2)


class MotifCaller(nn.Module):
    def __init__(self, hidden_size:int, n_layers:int, n_classes:int):
        super(MotifCaller, self).__init__()
        self.cnn_layers = ConvolutionalEncoder(hidden_size=hidden_size)
        self.rnn_layers = RNNLayers(
            hidden_size=hidden_size, num_layers=n_layers, n_classes=n_classes)

    def forward(self, x):
        x = self.cnn_layers.forward(x)
        x = self.rnn_layers.forward(x)
        return x


class NaiveCaller(nn.Module):
    def __init__(self, input_dim=1, conv_out=128, hidden_dim=128, num_layers=3, num_classes=5):
        
        super(NaiveCaller, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, stride=1, dilation=1),  
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=3),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=3),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(128, conv_out, kernel_size=5, stride=2, dilation=4),
            nn.ReLU()
            #nn.MaxPool1d(kernel_size=5, stride=4)  # Reduce sequence length
        )

        self.bigru = GRU(
            input_size=conv_out, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.2)
        
        self.fc = nn.Linear(hidden_dim * 2, num_classes) 

    def forward(self, x):
        
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bigru(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
    

class CallerSynthetic(nn.Module):
    def __init__(self, input_dim=1, conv_out=128, hidden_dim=256, num_layers=3, num_classes=17):
        
        super(NaiveCaller, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, stride=1, dilation=1),  
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=3),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(128, conv_out, kernel_size=5, stride=1, dilation=4),
            nn.ReLU()
            #nn.MaxPool1d(kernel_size=5, stride=4)  # Reduce sequence length
        )

        self.bigru = GRU(
            input_size=conv_out, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.2)
        
        self.fc = nn.Linear(hidden_dim * 2, num_classes) 

    def forward(self, x):
        
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bigru(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
    


class CallerEmpirical(nn.Module):
    def __init__(self, input_dim=1, conv_out=128, hidden_dim=256, num_layers=3, num_classes=19):
        super(NaiveCaller, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, stride=1, dilation=1),  
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=3),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=3),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(128, conv_out, kernel_size=5, stride=2, dilation=4),
            nn.ReLU()
            #nn.MaxPool1d(kernel_size=5, stride=4)  # Reduce sequence length
        )

        self.bigru = GRU(
            input_size=conv_out, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.2)
        
        self.fc = nn.Linear(hidden_dim * 2, num_classes) 

    def forward(self, x):
        
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bigru(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)