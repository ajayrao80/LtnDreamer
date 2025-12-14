import torch
import torch.nn as nn

"""
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
"""

class UpscaleNetwork(nn.Module):
    def __init__(self, stoch_dim=200):
        super().__init__()
        self.upscale_1 = nn.Sequential(
            nn.Linear(stoch_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128*14*14) 
        )

        self.upscale_2 = nn.Sequential(
            nn.Linear(stoch_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128*14*14) 
        )

        self.upscale_3 = nn.Sequential(
            nn.Linear(stoch_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128*14*14) 
        )

        #self.scalar_head = nn.Linear(200, 1)

    def forward(self, x):
        out_1 = self.upscale_1(x)
        out_1 = out_1.view(-1, 128, 14, 14)

        out_2 = self.upscale_1(x)
        out_2 = out_2.view(-1, 128, 14, 14)

        out_3 = self.upscale_1(x)
        out_3 = out_3.view(-1, 128, 14, 14)

        #scalar = self.scalar_head(x)
        return out_1, out_2, out_3 #, scalar
