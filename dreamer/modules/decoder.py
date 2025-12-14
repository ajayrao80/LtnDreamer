import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, embed_dim=1024, obs_shape=(3, 128, 128)):
        super().__init__()
        c, h, w = obs_shape
        self.fc = nn.Linear(embed_dim, 1024)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, 2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 6, 2), nn.ReLU(),
            nn.ConvTranspose2d(32, c, 2, 2), nn.Sigmoid() 
        )
    
    def forward(self, h):
        x = self.fc(h).view(-1, 1024, 1, 1)
        x = self.deconv(x)
        return x