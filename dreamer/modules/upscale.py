import torch
import torch.nn as nn

class UpscaleNetwork(nn.Module):
    def __init__(self, upscale_dim=3*128*128):
        super().__init__()
        self.fc = nn.Linear(200, upscale_dim) 

    def forward(self, x):
        x = self.fc(x)  
        x = x.view(-1, 3, 128, 128)
        return x
