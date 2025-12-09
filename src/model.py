import torch
import torch.nn as nn
import torch.nn.functional as F

class CarPricePredictor(nn.Module):
    
    def __init__(self, input_features, hidden_size=128, dropout_prob=0.5):
        super(CarPricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_features, hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.out = nn.Linear(hidden_size // 2, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.out(x)
        return x
