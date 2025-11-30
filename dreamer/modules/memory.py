import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicsModel(nn.Module):
    def __init__(self, embed_dim, logic_model, obs_shape=(3, 128, 128), action_dim=1):
        super().__init__()
        self.embed_dim = 128*14*14
        self.network = nn.Sequential(
            nn.Linear(6*self.embed_dim + action_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 6*self.embed_dim)
            )
        
        self.summary = nn.Sequential(
            nn.Linear(6*self.embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.embed_dim)
        )
        self.upscale_network = nn.Sequential(
            nn.Linear(self.embed_dim, 12500),
            nn.ReLU(),
            nn.Linear(12500, 25000),
            nn.ReLU(),
            nn.Linear(25000, obs_shape[0]*obs_shape[1]*obs_shape[2])
        )
        
        self.logic_model = logic_model

    def forward(self, prev_state, obs, action):
        front = self.logic_model.front(obs)
        right = self.logic_model.right(obs)
        up = self.logic_model.up(obs)
        prev_front, prev_right, prev_up = self.logic_model.front(prev_state), self.logic_model.right(prev_state), self.logic_model.up(prev_state)
        rot_change_front = self.logic_model.rot_change(front, action.unsqueeze(1))
        rot_change_right = self.logic_model.rot_change(right, action.unsqueeze(1))
        rot_change_up = self.logic_model.rot_change(up, action.unsqueeze(1))
        x = torch.cat([rot_change_front.flatten(start_dim=1), rot_change_right.flatten(start_dim=1), rot_change_up.flatten(start_dim=1), prev_front.flatten(start_dim=1), prev_right.flatten(start_dim=1), prev_up.flatten(start_dim=1), action.unsqueeze(1)], dim=-1)
        out = self.network(x)
        out = self.summary(out)
        out = self.upscale_network(out)
        out = out.view(-1, *obs.shape)
        return out.squeeze(0)
    


    


