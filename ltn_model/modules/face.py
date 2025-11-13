import torch
import torch.nn as nn

class Face(nn.Module):
    def __init__(self, in_channel=3, depth=32, ksize=4, stride=2):
        super().__init__()
        self.conv_network = nn.Sequential(
            nn.Conv2d(in_channel, depth*1, kernel_size=ksize, stride=stride),
            nn.ReLU(),
            nn.Conv2d(depth*1, depth*2, kernel_size=ksize, stride=stride),
            nn.ReLU(),
            nn.Conv2d(depth*2, depth*4, kernel_size=ksize, stride=stride),
            nn.ReLU()
        )
    
    def forward(self, x):
        out = self.conv_network(x)
        return out