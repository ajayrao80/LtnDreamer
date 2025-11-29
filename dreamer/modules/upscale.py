import torch
import torch.nn as nn

class UpscaleNetwork(nn.Module):
    def __init__(self, upscale_dim=3*128*128):
        super().__init__()
        self.upscale = nn.Sequential(
            nn.Linear(200, upscale_dim/4),
            nn.ReLU(),
            nn.Linear(upscale_dim/4, upscale_dim/2),
            nn.ReLU(),
            nn.Linear(upscale_dim/2, upscale_dim) 
        )

    def forward(self, x):
        x = self.upscale(x)
        x = x.view(-1, 3, 128, 128)
        return x
