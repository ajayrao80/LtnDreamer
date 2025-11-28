import torch
import torch.nn as nn

class Face(nn.Module):
    def __init__(self, in_channel=3, depth=32, ksize=4, stride=2, embed_dim=128*14*14):
        super().__init__()
        self.conv_network = nn.Sequential(
            nn.Conv2d(in_channel, depth*1, kernel_size=ksize, stride=stride),
            nn.ReLU(),
            nn.Conv2d(depth*1, depth*2, kernel_size=ksize, stride=stride),
            nn.ReLU(),
            nn.Conv2d(depth*2, depth*4, kernel_size=ksize, stride=stride),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(embed_dim, embed_dim)
        self.fc_logvar = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        out = self.conv_network(x)
        _, c, h, w = out.shape
        out = self.flatten(out)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        out = mu + eps * std
        return out.reshape(-1, c, h, w)
    
