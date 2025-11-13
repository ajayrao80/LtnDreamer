import torch
import torch.nn as nn

class RotQPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*14*14, 128*14*14)            
        )
    
    def forward(self, x):
        x = self.network(x)
        return x.view(-1, 128, 14, 14)

class RotQMinus(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*14*14, 128*14*14)            
        )
    
    def forward(self, x):
        x = self.network(x)
        return x.view(-1, 128, 14, 14)