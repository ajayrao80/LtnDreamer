import torch
import torch.nn as nn
import numpy as np
import random

from modules.rssm import RSSM
from modules.encoder import Encoder
from modules.decoder import Decoder

from utils.utils import (
    create_normal_dist,
    DynamicInfos,
)

#from dreamer.utils.buffer import ReplayBuffer

import wandb

class Dreamer:
    def __init__(self, _observation_shape, _stoch_size, _deter_size, _hidden_size, _embed_state_size, _action_size, _buffer, _batch_size=32, _ep_len=5, _train_iterations=1000, _lr=0.0006, _device="cuda", _login_key=None):
        self.device = _device
        self.buffer = _buffer #ReplayBuffer(_observation_shape, _action_size)
        self.lr = _lr
        self.action_size = _action_size
        self.login_key = _login_key

        self.encoder = Encoder(_observation_shape).to(self.device)
        self.decoder = Decoder(_observation_shape, _stoch_size, _deter_size).to(self.device)
        self.rssm = RSSM(_stoch_size, _deter_size, _hidden_size, _embed_state_size, _action_size, self.device).to(self.device)        

        self.model_params = (
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.rssm.parameters())
        )

        self.model_optimizer = torch.optim.Adam(self.model_params, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.model_optimizer, mode='min', factor=0.1, patience=10)
        self.dynamic_learning_infos = DynamicInfos(self.device)

        self.collect_interval = 500
        self.train_iterations = _train_iterations
        self.batch_size = _batch_size
        self.ep_len = _ep_len
        self.free_nats = 3
        self.kl_divergence_scale = 1
        self.clip_grad = 100
        self.grad_norm_type = 2
        self.num_interaction_episodes = 0
        self.seed_episodes = 1000
    
    def log(self, loss, post, deterministic):
        print(f"Loss: {loss}")
        print("Posterior reconstruction")
        with torch.no_grad():
            image = self.decoder(post, deterministic).sample()[1][1]
            wandb.log({"reconstruction": wandb.Image(image)})
            wandb.log({"loss": loss})

    def train(self, env=None):
        wandb.login(key=self.login_key)
        wandb.init(project="vanilla_world_model")

        if len(self.buffer) < 1:
            self.environment_interaction(env, self.seed_episodes)

        for iteration in range(self.train_iterations):
            loss = 0
            posteriors, deterministics = None, None
            for collect_interval in range(self.collect_interval):
                data = self.buffer.sample(
                    self.batch_size, self.ep_len
                )

                posteriors, deterministics, _loss = self.dynamic_learning(data)
                loss += (_loss)/self.batch_size

            #self.environment_interaction(env, self.num_interaction_episodes)
            # todo: evaluate
            #self.scheduler.step(loss/self.collect_interval)
            self.log(loss/self.collect_interval, posteriors, deterministics)
        
        wandb.finish()
    
    def dynamic_learning(self, data):
        prior, deterministic = self.rssm.recurrent_model_input_init(len(data.action))

        data.embedded_observation = self.encoder(data.observation)

        for t in range(1, self.ep_len):
            deterministic = self.rssm.recurrent_model(
                prior, data.action[:, t-1], deterministic
            )

            prior_dist, prior = self.rssm.transition_model(deterministic)
            posterior_dist, posterior = self.rssm.representation_model(
                data.embedded_observation[:, t], deterministic
            )

            self.dynamic_learning_infos.append(
                priors=prior,
                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                posteriors=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
            )

            prior = posterior
        
        infos = self.dynamic_learning_infos.get_stacked()
        loss = self._model_update(data, infos)

        infos_derm = infos.deterministics.detach()
        infos_post = infos.posteriors.detach()
        
        return infos_post, infos_derm, loss
    
    def _model_update(self, data, posterior_info):
        reconstructed_observation_dist = self.decoder(
            posterior_info.posteriors, posterior_info.deterministics
        )

        reconstruction_observation_loss = reconstructed_observation_dist.log_prob(
            data.observation[:, 1:]
        )

        prior_dist = create_normal_dist(
            posterior_info.prior_dist_means,
            posterior_info.prior_dist_stds,
            event_shape=1,
        )
        posterior_dist = create_normal_dist(
            posterior_info.posterior_dist_means,
            posterior_info.posterior_dist_stds,
            event_shape=1,
        )
        kl_divergence_loss = torch.mean(
            torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
        )
        kl_divergence_loss = torch.max(
            torch.tensor(self.free_nats).to(self.device), kl_divergence_loss
        )
        model_loss = (
            self.kl_divergence_scale * kl_divergence_loss
            - reconstruction_observation_loss.mean()
        )

        self.model_optimizer.zero_grad()
        model_loss.backward()
        l = model_loss.item()
        nn.utils.clip_grad_norm_(
            self.model_params,
            self.clip_grad,
            norm_type=self.grad_norm_type,
        )
        self.model_optimizer.step()

        return l
    
    @torch.no_grad()
    def environment_interaction(self, env, num_interaction_episodes, train=True):
        for epi in range(num_interaction_episodes):
            posterior, deterministic = self.rssm.recurrent_model_input_init(1)
            action = [0, 0, 0, 0, 0, 0, 0] #torch.zeros(1, self.action_size)

            observation = env.reset()[0]
            embedded_observation = self.encoder(
                torch.from_numpy(observation).float().to(self.device)  
            )

            done = False

            steps = 0
            while not done:
                deterministic = self.rssm.recurrent_model(
                    posterior, torch.tensor(action).unsqueeze(0).to(self.device), deterministic
                )

                embedded_observation = embedded_observation.reshape(1, -1)
                _, posterior = self.rssm.representation_model(
                    embedded_observation, deterministic
                )

                action_idx = random.randrange(0, 6)
                action = [-1, -1, -1, -1, action_idx, -1, -1]

                next_state, _, done, _ = env.step(action)

                if train:
                    self.buffer.add(
                        observation, np.array(action), next_state[0]
                    )

                embedded_observation = self.encoder(
                    torch.from_numpy(next_state[0]).float().to(self.device)
                )

                observation = next_state[0]

                steps += 1

                if steps >= self.ep_len:
                    observation = env.reset()[0]
                    done = True