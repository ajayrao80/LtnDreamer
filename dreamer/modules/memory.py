import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicsModel(nn.Module):
    def __init__(self, embed_dim, logic_model, obs_shape=(3, 128, 128), action_dim=1):
        super().__init__()
        self.embed_dim = 128*14*14
        self.network = nn.Sequential(
            nn.Linear(3*self.embed_dim + action_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 3*self.embed_dim)
            )
        
        self.summary = nn.Sequential(
            nn.Linear(3*self.embed_dim, 1024),
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

    def forward(self, obs, action):
        front = self.logic_model.front(obs)
        right = self.logic_model.right(obs)
        up = self.logic_model.up(obs)
        #rot_change_front = self.logic_model.rot_change(front, action)
        #rot_change_right = self.logic_model.rot_change(right, action)
        #rot_change_up = self.logic_model.rot_change(up, action)
        x = torch.cat([front.flatten(start_dim=1), right.flatten(start_dim=1), up.flatten(start_dim=1), action], dim=-1)
        out = self.network(x)
        out = self.summary(out)
        out = self.upscale_network(out)
        out = out.view(-1, *obs.shape)
        return out.squeeze(0)
    


    


