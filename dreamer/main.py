import argparse
from dataset.dataset import Dataset
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.model import RSSM
import torch
from dataset.dataset import Dataset
from utils.utils import log
import wandb

def eval_rollout(dataset, encoder, rssm, decoder, T=5):
    with torch.no_grad():
        sample = dataset.sample(1, T)
        initial_obs = sample.observation[0, 0].unsqueeze(0)
        action_seq = sample.action
        # Get embedding from encoder
        embed = encoder(initial_obs)  # [B, embed_dim]

        B = initial_obs.size(0)
        device = initial_obs.device
        deter = torch.zeros(B, rssm.rnn.hidden_size, device=device)
        stoch = torch.zeros(B, rssm.fc_prior.out_features // 2, device=device)
        # One step with observation: use posterior (with obs)
        _, _, _, post_stoch, _, _, deter = rssm(stoch, deter, torch.zeros(B, action_seq.size(-1), device=device), embed)  
        
        # rollout
        stoch_state = post_stoch
        deter_state = deter

        ground_truth_images = []
        reconstructed_images = []

        for t in range(1, action_seq.size(1)):
            #wandb.log({"Ground Truth": wandb.Image(sample.observation[0, t]) })
            #wandb.log({"Reconstruction": wandb.Image(decoder(stoch_state)[0]) })
            ground_truth_images.append(wandb.Image(sample.observation[0, t]))
            reconstructed_images.append(wandb.Image(decoder(stoch_state)[0]))

            prior_stoch, prior_mean, prior_std, _, _, _, deter_state = rssm(
                stoch_state, deter_state, action_seq[:, t-1], embed=None  # embed=None: imagination
            )
            stoch_state = prior_stoch
        
        wandb.log({"Ground Truth": ground_truth_images})
        wandb.log({"Imagination": reconstructed_images})
        

def main(lr, epochs, embed_dim, stoch_dim, deter_dim, dataset_train_path, dataset_test_path, beta, login_key, free_nats=3.0):
    obs_shape = (3, 128, 128)
    action_dim = 7
    embed_dim = embed_dim
    stoch_dim = stoch_dim
    deter_dim = deter_dim

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    encoder = Encoder(obs_shape, embed_dim).to(device)
    decoder = Decoder(embed_dim, obs_shape).to(device)
    rssm = RSSM(action_dim, stoch_dim, deter_dim, embed_dim).to(device)
    optim_model = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(rssm.parameters()), lr=lr)
    dataset_object = Dataset(obs_shape, action_dim, device, dataset_train_path, dataset_test_path)
    dataset_train = dataset_object.get_dataset_train()
    dataset_test = dataset_object.get_dataset_test()

    B, T = 32, 5
    total_iterations = 750
    epochs = epochs
    beta = beta
    login_key = login_key

    wandb.login(key=login_key)
    wandb.init(project="vanilla_world_model")

    for epoch in range(epochs): 
        for iteration in range(total_iterations):
            deter = torch.zeros(B, deter_dim, device=device)
            stoch = torch.zeros(B, stoch_dim, device=device)
            kld_loss = 0.
            recon_loss = 0.

            for t in range(1, T):
                sample = dataset_train.sample(B, T)
                obs = sample.observation 
                actions = sample.action        
                embed = encoder(obs[:, t])
                prior_stoch, prior_mean, prior_std, post_stoch, post_mean, post_std, deter = rssm(stoch, deter, actions[:, t-1], embed)
                recon_mean = decoder(post_stoch)
                fixed_std = 1.0
                dist = torch.distributions.Normal(recon_mean, fixed_std)
                recon_log_prob = dist.log_prob(obs[:, t]).sum(dim=[1,2,3]).mean()
                recon_loss += -recon_log_prob

                kld = torch.distributions.kl_divergence(
                    torch.distributions.Normal(post_mean, post_std),
                    torch.distributions.Normal(prior_mean, prior_std)
                ).mean()

                kld = torch.max(
                    torch.tensor(free_nats).to(device), kld
                )

                kld_loss += kld
                stoch = post_stoch
            
            loss = recon_loss + kld_loss * beta
            optim_model.zero_grad()
            loss.backward()
            optim_model.step()
   
        wandb.log({"Epoch": epoch})
        wandb.log({"Reconstruction Loss": recon_loss.item()})
        wandb.log({"KLD Loss": kld_loss.item()})
        eval_rollout(dataset_test, encoder, rssm, decoder)
        print(f"Epoch {epoch}: recon_loss={recon_loss.item():.2f}, kld_loss={kld_loss.item():.2f}")
    
    wandb.finish()



