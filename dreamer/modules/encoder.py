import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, _observation_shape, embed_dim=1024):
        super().__init__()
        c, h, w = _observation_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2), nn.ReLU()
        )

        self.fc = nn.Linear(256*6*6, embed_dim)
    
    def forward(self, obs):
        x = self.encoder(obs)
        x = x.view(x.size(0), -1)
        return self.fc(x)