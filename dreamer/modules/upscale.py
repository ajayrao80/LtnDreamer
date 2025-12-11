import torch
import torch.nn as nn

class UpscaleNetwork(nn.Module):
    def __init__(self, upscale_dim=3*128*128):
        super().__init__()
        self.upscale = nn.Sequential(
            nn.Linear(200, 12500),
            nn.ReLU(),
            nn.Linear(12500, 25000),
            nn.ReLU(),
            nn.Linear(25000, upscale_dim) 
        )

        #self.scalar_head = nn.Linear(200, 1)

    def forward(self, x):
        out = self.upscale(x)
        out = out.view(-1, 3, 128, 128)

        #scalar = self.scalar_head(x)
        return out #, scalar
