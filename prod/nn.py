
import torch.nn as nn
from torch.nn import GRU
import torch.nn.functional as F


class CallerEmpirical(nn.Module):
    def __init__(
            self, input_dim=1, conv_out=128, hidden_dim=256, num_layers=3, num_classes=19):
        
        super(CallerEmpirical, self).__init__()
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
