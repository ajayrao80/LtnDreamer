import torch
import torch.nn as nn

from utils.utils import (
    initialize_weights,
    horizontal_forward,
)

class Encoder(nn.Module):
    def __init__(self, observation_shape, _depth=32, _kernel_size=4, _stride=2):
        super().__init__()
        self.observation_shape = observation_shape
        self.depth = _depth
        self.kernel_size = _kernel_size
        self.stride = _stride

        self.network = nn.Sequential(
            nn.Conv2d(
                self.observation_shape[0],
                self.depth * 1,
                self.kernel_size,
                self.stride
            ),
            nn.ReLU(),

            nn.Conv2d(
                self.depth * 1,
                self.depth * 2,
                self.kernel_size,
                self.stride
            ),
            nn.ReLU(),

            nn.Conv2d(
                self.depth * 2,
                self.depth * 4,
                self.kernel_size,
                self.stride
            ),
            nn.ReLU(),

            nn.Conv2d(
                self.depth * 4,
                self.depth * 8,
                self.kernel_size,
                self.stride
            ),
            nn.ReLU()
        )
        
        self.network.apply(initialize_weights)

    def forward(self, x):
        return horizontal_forward(self.network, x, input_shape=self.observation_shape)
    
