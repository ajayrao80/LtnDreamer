import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from utils.utils import create_normal_dist, build_network, horizontal_forward

class RSSM(nn.Module):
    def __init__(self, _stochastic_size, _deter_size, _hidden_size, _embed_state_size, _action_size, _device):
        super().__init__()
        
        self.device = _device
        self.recurrent_model = RecurrentModel(_stochastic_size, _deter_size, _hidden_size, _action_size, self.device)
        self.transition_model = TransitionModel(_stochastic_size, _deter_size, _hidden_size, self.device) # prior
        self.representation_model = RepresentationModel(_embed_state_size, _stochastic_size, _deter_size, _hidden_size, self.device)
    
    def recurrent_model_input_init(self, batch_size):
        return self.transition_model.input_init(batch_size), self.recurrent_model.input_init(batch_size)
        

class RecurrentModel(nn.Module):
    def __init__(self, _stochastic_size, _deter_size, _hidden_size, _action_size, _device):
        super().__init__()
        self.stochastic_size = _stochastic_size
        self.hidden_size = _hidden_size
        self.action_size = _action_size
        self.determ_size = _deter_size
        self.device = _device

        self.activation = nn.ELU()
        self.linear = nn.Linear(
            self.stochastic_size + self.action_size, self.hidden_size 
        )

        self.recurrent = nn.GRUCell(self.hidden_size, self.determ_size)
    
    def forward(self, embedded_state, action, deterministic):
        x = torch.cat([embedded_state, action], 1)
        x = self.activation(self.linear(x))
        x = self.recurrent(x, deterministic)
        return x
    
    def input_init(self, batch_size):
        return torch.zeros(batch_size, self.determ_size).to(self.device)

class TransitionModel(nn.Module):
    def __init__(self, _stochastic_size, _deter_size, _hidden_size, _device, _num_layers=2, _act="ELU", _min_std=0.1):
        super().__init__()
        self.device = _device
        self.stoch_size = _stochastic_size
        self.determ_size = _deter_size
        self.hidden_size = _hidden_size
        self.num_layers = _num_layers
        self.activation = _act
        self.min_std = _min_std

        self.network = build_network(
            self.determ_size, 
            self.hidden_size,
            self.num_layers,
            self.activation,
            self.stoch_size * 2            
        )
    
    def forward(self, x):
        x = self.network(x)
        prior_dist = create_normal_dist(x, min_std=self.min_std)
        prior = prior_dist.rsample()
        return prior_dist, prior
    
    def input_init(self, batch_size):
        return torch.zeros(batch_size, self.stoch_size).to(self.device)

class RepresentationModel(nn.Module):
    def __init__(self, _embed_state_size, _stoch_size, _determ_size, _hidden_size, _device, _num_layers=2, _act="ELU", _min_std=0.1):
        super().__init__()
        self.embed_state_size = _embed_state_size
        self.stoch_size = _stoch_size
        self.determ_size = _determ_size
        self.hidden_size = _hidden_size
        self.num_layers = _num_layers
        self.activation = _act
        self.min_std = _min_std
        self.device = _device

        self.network = build_network(
            self.embed_state_size + self.determ_size,
            self.hidden_size,
            self.num_layers,
            self.activation,
            self.stoch_size * 2
        )
    
    def forward(self, embedded_obs, deterministic):
        x = self.network(torch.cat((embedded_obs, deterministic), 1))
        posterior_dist = create_normal_dist(x, min_std=self.min_std)
        posterior = posterior_dist.rsample()
        return posterior_dist, posterior