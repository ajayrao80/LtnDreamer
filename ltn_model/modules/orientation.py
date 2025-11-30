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
    
class RotChange(nn.Module):
    def __init__(self, action_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*14*14 + action_dim, 128*14*14)
        )   

    def forward(self, x, action):
        x = self.network(torch.cat([x.view(x.size(0), -1), action], dim=-1))
        return x.view(-1, 128, 14, 14)
    