import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, depth=32, ksize=4, stride=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(depth*4*3, depth*2, kernel_size=ksize, stride=stride),
            nn.ReLU(),
            nn.ConvTranspose2d(depth*2, depth*1, kernel_size=ksize+1, stride=stride),
            nn.ReLU(),
            nn.ConvTranspose2d(depth*1, 3, kernel_size=ksize, stride=stride),
            nn.Sigmoid()
        )
    
    def forward(self, fx, rx, ux):
        x = torch.stack([fx, rx, ux])
        x = x.permute(1, 0, 2, 3, 4)
        out = self.network(x.reshape(-1, x.shape[1]*x.shape[2], x.shape[3], x.shape[4]))
        return out
    
    