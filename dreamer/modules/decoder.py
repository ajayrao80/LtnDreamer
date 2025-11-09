import torch.nn as nn

from utils.utils import (
    initialize_weights,
    horizontal_forward,
    create_normal_dist,
)

class Decoder(nn.Module):
    def __init__(self, _observation_shape, _stoch_size, _deter_size, _depth=32, _kernel_size=4, _stride=2):
        super().__init__()
        self.stochastic_size = _stoch_size
        self.deterministic_size = _deter_size
        self.observation_shape = _observation_shape
        self.depth = _depth
        self.kernel_size = _kernel_size
        self.stride = _stride

        self.network = nn.Sequential(
            nn.Linear(
                self.deterministic_size + self.stochastic_size, self.depth * 32
            ),
            nn.Unflatten(1, (self.depth * 32, 1)),
            nn.Unflatten(2, (1, 1)),

            nn.ConvTranspose2d(
                self.depth * 32,
                self.depth * 16,
                self.kernel_size + 1,
                self.stride
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                self.depth * 16,
                self.depth * 8,
                self.kernel_size + 1,
                self.stride
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                self.depth * 8,
                self.depth * 4,
                self.kernel_size + 2,
                self.stride
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                self.depth * 4,
                self.depth * 2,
                self.kernel_size + 1,
                self.stride
            ),

            nn.ReLU(),
            nn.ConvTranspose2d(
                self.depth * 2,
                self.observation_shape[0],
                self.kernel_size,
                self.stride
            ),
            nn.Sigmoid()
        )
        
        self.network.apply(initialize_weights)
    
    def forward(self, posterior, deterministic):
        x = horizontal_forward(
            self.network, posterior, deterministic, output_shape=self.observation_shape
        )

        dist = create_normal_dist(x, std=1, event_shape=len(self.observation_shape))
        return dist