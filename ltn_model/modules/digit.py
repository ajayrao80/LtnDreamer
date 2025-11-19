import torch
import torch.nn as nn

class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*14*14, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.Softmax()
        )
    
    def forward(self, x, d=None):
        out = self.network(x)
        if d is not None:
            out = torch.sum(out*d, dim=1)
        return out