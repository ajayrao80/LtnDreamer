def attrdict_monkeypatch_fix():
    import collections
    import collections.abc
    for type_name in collections.abc.__all__:
            setattr(collections, type_name, getattr(collections.abc, type_name))

attrdict_monkeypatch_fix()

from attrdict import AttrDict
import torch
import numpy as np

class ReplayBuffer(object):
    def __init__(self, observation_shape, action_size, device="cuda", capacity=25000):
        self.device = device
        self.capacity = int(capacity)

        state_type = np.uint8 if len(observation_shape) < 3 else np.float32

        self.observation = np.empty(
            (self.capacity, *observation_shape), dtype=state_type
        )
        #self.next_observation = np.empty(
        #    (self.capacity, *observation_shape), dtype=state_type
        #)
        self.action = np.empty((self.capacity, action_size), dtype=np.float32)
        #self.reward = np.empty((self.capacity, 1), dtype=np.float32)
        #self.done = np.empty((self.capacity, 1), dtype=np.float32)

        self.buffer_index = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.buffer_index

    def add(self, observation, action):
        self.observation[self.buffer_index] = observation
        self.action[self.buffer_index] = action
        #self.reward[self.buffer_index] = reward
        #self.next_observation[self.buffer_index] = next_observation
        #self.done[self.buffer_index] = done

        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.full = self.full or self.buffer_index == 0
    
    def get(self, index):
        sample = AttrDict(
            {
                "observation": torch.as_tensor(self.observation[index]),
                "action": torch.as_tensor(self.action[index])
                #"next_observation": torch.as_tensor(self.next_observation[index])
            }
        )

        return sample

    def sample(self, batch_size, chunk_size):
        """
        (batch_size, chunk_size, input_size)
        """
        last_filled_index = self.buffer_index - chunk_size + 1
        assert self.full or (
            last_filled_index > batch_size
        ), "too short dataset or too long chunk_size"

        valid_indices = np.arange(0, self.capacity if self.full else last_filled_index, 5)
        
        #sample_index = np.random.randint(
        #    0, self.capacity if self.full else last_filled_index, batch_size
        #).reshape(-1, 1)
        sample_index = np.random.randint(
            0, len(valid_indices), batch_size
        ).reshape(-1, 1)
        sample_index = valid_indices[sample_index]

        chunk_length = np.arange(chunk_size).reshape(1, -1)

        sample_index = (sample_index + chunk_length) % self.capacity

        #print(sample_index)

        observation = torch.as_tensor(
            self.observation[sample_index], device=self.device
        ).float()
        #next_observation = torch.as_tensor(
        #    self.next_observation[sample_index], device=self.device
        #).float()

        action = torch.as_tensor(self.action[sample_index], device=self.device)
        #reward = torch.as_tensor(self.reward[sample_index], device=self.device)
        #done = torch.as_tensor(self.done[sample_index], device=self.device)

        sample = AttrDict(
            {
                "observation": observation,
                "action": action
                #"reward": reward,
                #"next_observation": next_observation,
                #"done": done,
            }
        )
        return sample

def log(sample_obs, encoder, decoder, rssm, deter_dim, stoch_dim, action_dim, device):
    with torch.no_grad():
        sample_obs = sample_obs.observation[0, 2].unsqueeze(0)  # Shape: [1, C, H, W]
        embed = encoder(sample_obs)

        deter = torch.zeros(1, deter_dim, device=device)
        stoch = torch.zeros(1, stoch_dim, device=device)
        prior_stoch, _, _, post_stoch, _, _, _ = rssm(stoch, deter, torch.zeros(1, action_dim, device=device), embed)
        recon = decoder(prior_stoch)
        recon_post = decoder(post_stoch)

        #original_img = sample_obs.cpu().squeeze().permute(1,2,0).numpy()
        #recon_img = recon.cpu().squeeze().permute(1,2,0).numpy()
        #recon_img_post = recon_post.cpu().squeeze().permute(1,2,0).numpy()

        return recon, recon_post

        #original_img = original_img.clip(0, 1)
        #recon_img = recon_img.clip(0, 1)

        #show(recon_img) # prior
        #show(recon_img_post)
        #show(original_img)

