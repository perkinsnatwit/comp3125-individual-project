import torch
import torch.nn as nn
import torch.nn.functional as F

class CarPricePredictor(nn.Module):
    
    def __init__(self, input_features, hidden_size=128):
        super(CarPricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
