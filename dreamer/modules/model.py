import torch
import torch.nn as nn
import torch.nn.functional as F

class RSSM(nn.Module):
    def __init__(self, action_dim=7, stoch_dim=60, determ_dim=400, embed_dim=1024):
        super().__init__()
        self.rnn= nn.GRUCell(stoch_dim + action_dim, determ_dim)
        self.fc_prior = nn.Linear(determ_dim, 2*stoch_dim)
        self.fc_post = nn.Linear(determ_dim + embed_dim, 2 * stoch_dim)
    
    def forward(self, prev_stoch, prev_deter, action, embed=None):
        x = torch.cat([prev_stoch, action], -1)
        deter = self.rnn(x, prev_deter)

        prior_stats = self.fc_prior(deter)
        prior_mean, prior_std = torch.chunk(prior_stats, 2, -1)
        prior_std = F.softplus(prior_std) + 1e-4
        prior_stoch = prior_mean + prior_std * torch.randn_like(prior_mean)

        if embed is not None:
            post_in = torch.cat([deter, embed], -1)
            post_stats = self.fc_post(post_in)
            post_mean, post_std = torch.chunk(post_stats, 2, -1)
            post_std = F.softplus(post_std) + 1e-4
            post_stoch = post_mean + post_std * torch.randn_like(post_mean)
        else:
            post_stoch, post_mean, post_std = prior_stoch, prior_mean, prior_std
        
        return prior_stoch, prior_mean, prior_std, post_stoch, post_mean, post_std, deter
    
    def imagine(self, actions, horinzon, device):
        B = actions.size(0)
        deter = torch.zeros(B, self.rnn.hidden_size, device=device)
        stoch = torch.zeros(B, self.fc_prior.out_features // 2, device=device)
        priors = []

        for t in range(horinzon):
            prior_stoch, prior_mean, prior_std, _, _, _, deter = self.forward(stoch, deter, actions[:, t], embed=None)
            priors.append(prior_stoch)
            stoch = prior_stoch
        
        return torch.stack(priors, dim=1)
    


