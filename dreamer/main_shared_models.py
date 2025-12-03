import argparse
from dreamer.dataset.dataset import Dataset
from dreamer.modules.encoder import Encoder
from dreamer.modules.decoder import Decoder
from dreamer.modules.model import RSSM
from dreamer.modules.upscale import UpscaleNetwork
import torch
from dreamer.utils.utils import log
import wandb
from dreamer.utils.utils import save_model
from ltn_model.ltn_qm.logic_loss import LogicLoss

def eval_loss(dataset, encoder, rssm, decoder, logic_loss_object, upscale_network, T=5, batch_size=32):
    logic_loss_object.ltn_models.front.eval()
    logic_loss_object.ltn_models.right.eval()
    logic_loss_object.ltn_models.up.eval() #encoder.eval()
    decoder.eval()
    rssm.eval()
    sample = dataset.sample(batch_size, T)
    obs = sample.observation
    actions = sample.action
    with torch.no_grad():
        B, T = obs.shape[0], obs.shape[1]
        deter_dim = rssm.rnn.hidden_size
        stoch_dim = rssm.fc_prior.out_features // 2

        recon_loss_total = 0.
        kld_loss_total = 0.
        logic_loss_total = 0.

        device = obs.device

        deter = torch.zeros(B, deter_dim, device=device)
        stoch = torch.zeros(B, stoch_dim, device=device)

        for t in range(1, T):
            embed = encoder(obs[:, t-1], logic_loss_object.ltn_models) 
            prior_stoch, prior_mean, prior_std, post_stoch, post_mean, post_std, deter = rssm(
                stoch, deter, actions[:, t-1], embed)
            
            upscaled_post_stoch, scalar = upscale_network(post_stoch)
            mean = decoder(logic_loss_object.ltn_models.front(upscaled_post_stoch), logic_loss_object.ltn_models.right(upscaled_post_stoch), logic_loss_object.ltn_models.up(upscaled_post_stoch)) #mean = decoder(post_stoch)
            dist = torch.distributions.Normal(mean, 1.0)  # Decoder returns mean
            log_prob = -dist.log_prob(obs[:, t]).sum(dim=[1,2,3]).mean()  # Per batch
            
            actions_batch = actions[:, t-1].max(dim=1, keepdim=True).values.squeeze(1)
            logic_loss = logic_loss_object.compute_logic_loss(obs[:, t-1], actions_batch, mean) if logic_loss_object is not None else "-"

            kld = torch.distributions.kl_divergence(
                torch.distributions.Normal(post_mean, post_std),
                torch.distributions.Normal(prior_mean, prior_std)
            ).mean()

            kld_loss_total += kld
            recon_loss_total += log_prob
            logic_loss_total = logic_loss_total+logic_loss if logic_loss_object is not None else "-"

            stoch = post_stoch

        metrics = {
            'reconstruction_logprob': recon_loss_total.item(),
            'kl_loss': kld_loss_total.item(),
            'logic_loss': logic_loss_total.item() * recon_loss_total.item()
        }

    return metrics

def eval_rollout(dataset, encoder, rssm, decoder, logic_loss_object, upscale_network, T=5):
    with torch.no_grad():
        sample = dataset.sample(1, T)
        initial_obs = sample.observation[0, 0].unsqueeze(0)
        action_seq = sample.action
        # Get embedding from encoder
        embed = encoder(initial_obs, logic_loss_object.ltn_models)  # [B, embed_dim]

        B = initial_obs.size(0)
        device = initial_obs.device
        deter = torch.zeros(B, rssm.rnn.hidden_size, device=device)
        stoch = torch.zeros(B, rssm.fc_prior.out_features // 2, device=device)
        # One step with observation: use posterior (with obs)
        #_, _, _, post_stoch, _, _, deter = rssm(stoch, deter, torch.zeros(B, action_seq.size(-1), device=device), embed)  
        
        # rollout
        stoch_state = stoch #post_stoch
        deter_state = deter

        ground_truth_images = [wandb.Image(sample.observation[0, 0])]
        reconstructed_images = []
        
        for t in range(1, T):
            prior_stoch, prior_mean, prior_std, post_stoch, post_mean, post_std, deter_state = rssm(
                stoch_state, deter_state, action_seq[:, t-1], embed=embed  
            )

            upscaled_post_stoch, scalar = upscale_network(post_stoch)
            reconstructed_image = decoder(logic_loss_object.ltn_models.front(upscaled_post_stoch), logic_loss_object.ltn_models.right(upscaled_post_stoch), logic_loss_object.ltn_models.up(upscaled_post_stoch))
            #reconstructed_image = decoder(post_stoch)

            ground_truth_images.append(wandb.Image(sample.observation[0, t]))
            reconstructed_images.append(wandb.Image(reconstructed_image[0]))

            stoch_state = post_stoch
            embed=None

        roll_outs = {
            "Ground Truth": ground_truth_images,
            "Imagination": reconstructed_images
        }

        return roll_outs
    
def encoder(obs, logic_models):
    front = logic_models.front(obs)
    right = logic_models.right(obs)
    up = logic_models.up(obs)
    return torch.cat([front, right, up], dim=1).flatten(1)

def get_ltn_predictions(dataset, logic_loss_object, T=5):
    with torch.no_grad():
        sample = dataset.sample(1, T)
        initial_obs = sample.observation[0, 0].unsqueeze(0)
        action = sample.action[0, 0].unsqueeze(0)
        action = action.max(dim=1, keepdim=True).values.squeeze(1)

        ltn_reconstruction_pred = logic_loss_object.get_ltn_predictions(initial_obs, action)
        return {"LTN Reconstruction": wandb.Image(ltn_reconstruction_pred[0]), "Ground Truth": wandb.Image(sample.observation[0, 1])}

def main(lr, epochs, embed_dim, stoch_dim, deter_dim, dataset_train_path, dataset_test_path, beta, login_key, model_save_path, logic_models_path=None, free_nats=3.0, project_name="vanilla_world_model", logic_weight=100.0, logic_decay_rate=1.0, train_all=True, batch_size=32):
    obs_shape = (3, 128, 128)
    action_dim = 7
    embed_dim = embed_dim
    stoch_dim = stoch_dim
    deter_dim = deter_dim

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    #encoder = Encoder(obs_shape, embed_dim).to(device)
    #decoder = Decoder(stoch_dim, obs_shape).to(device) #Decoder(embed_dim, obs_shape).to(device)
    rssm = RSSM(action_dim, stoch_dim, deter_dim, embed_dim).to(device)
    upscale_network = UpscaleNetwork(upscale_dim=obs_shape[0]*obs_shape[1]*obs_shape[2]).to(device)
    dataset_object = Dataset(obs_shape, action_dim, device, dataset_train_path, dataset_test_path)
    dataset_train = dataset_object.get_dataset_train()
    dataset_test = dataset_object.get_dataset_test()

    B, T = batch_size, 5
    total_iterations = int(dataset_train.observation.shape[0] / B)
    epochs = epochs
    beta = beta
    #login_key = login_key

    logic_loss_object = None
    if logic_models_path is not None:
        logic_loss_object = LogicLoss(logic_models_path, model_name_digits=None, train_all=train_all)
    
    decoder = logic_loss_object.ltn_models.dec if logic_loss_object is not None else Decoder(stoch_dim, obs_shape).to(device)
    
    if not train_all:
        optim_model = torch.optim.Adam(list(rssm.parameters()), lr=lr) # list(decoder.parameters()) + #list(encoder.parameters()) + 
    else:
        optim_model = torch.optim.Adam(list(upscale_network.parameters()) + list(rssm.parameters()) + logic_loss_object.get_logic_parameters(), lr=lr) #list(encoder.parameters()) + list(decoder.parameters())

    wandb.login(key=login_key)
    wandb.init(project=project_name)

    for epoch in range(epochs): 
        l = 0.
        rl = 0.
        kld_l = 0.
        logic_l = 0.

        for iteration in range(total_iterations):
            deter = torch.zeros(B, deter_dim, device=device)
            stoch = torch.zeros(B, stoch_dim, device=device)
            
            kld_loss = 0.
            recon_loss = 0.
            logic_loss_total = 0.
            
            sample = dataset_train.sample(B, T)
            obs = sample.observation
            actions = sample.action
                
            sample = dataset_train.sample(B, T)
            obs = sample.observation    
            for t in range(1, T):
                actions = sample.action
                embed = encoder(obs[:, t-1], logic_loss_object.ltn_models)  # Updated encoder with logic models
                prior_stoch, prior_mean, prior_std, post_stoch, post_mean, post_std, deter = rssm(stoch, deter, actions[:, t-1], embed)
                upscaled_post_stoch, scalar = upscale_network(post_stoch)
                recon_mean = decoder(logic_loss_object.ltn_models.front(upscaled_post_stoch), logic_loss_object.ltn_models.right(upscaled_post_stoch), logic_loss_object.ltn_models.up(upscaled_post_stoch)) #recon_mean = decoder(post_stoch)
                """
                fixed_std = 1.0
                dist = torch.distributions.Normal(recon_mean, fixed_std)
                recon_log_prob = dist.log_prob(obs[:, t]).sum(dim=[1,2,3]).mean()
                recon_loss += -recon_log_prob
                """
                
                actions_batch = actions[:, t-1].max(dim=1, keepdim=True).values.squeeze(1)
                ltn_loss = logic_loss_object.compute_logic_loss(obs[:, t-1], actions_batch, obs[:, t]) if train_all else 0.

                #if epoch >= epochs//2:
                logic_loss = logic_loss_object.compute_logic_loss(obs[:, t-1], actions_batch, recon_mean) if logic_models_path is not None else 0.
                #else:
                #    logic_loss = 0.
                
                logic_loss_total += ltn_loss + logic_loss 
                #print(f"Logic Loss: {logic_loss}, Logic Loss Total:{logic_loss_total}")
                
                kld = torch.distributions.kl_divergence(
                    torch.distributions.Normal(post_mean, post_std),
                    torch.distributions.Normal(prior_mean, prior_std)
                ).mean()

                kld = torch.max(
                    torch.tensor(free_nats).to(device), kld
                )

                kld_loss += kld
                stoch = post_stoch

            #logic_weight = recon_loss.item()
            logic_loss_total = logic_weight*logic_loss_total
            kld_loss = (kld_loss * beta)
            loss = kld_loss + logic_loss_total # recon_loss
            optim_model.zero_grad()
            loss.backward()
            optim_model.step()

            l += loss.item()
            logic_l += logic_loss_total.item()
            rl += recon_loss.item()
            kld_l += kld_loss.item()
        
        rollout_metrics = eval_rollout(dataset_test, encoder, rssm, decoder, logic_loss_object, upscale_network)
        loss_metrics = eval_loss(dataset_test, encoder, rssm, decoder, logic_loss_object, upscale_network)
        ltn_predictions = get_ltn_predictions(dataset_test, logic_loss_object)

        metrics = {
            "Epoch": epoch,
            "Loss": l/total_iterations,
            #"Reconstruction Loss Train": rl/total_iterations,
            "Logic Loss Train": logic_l/total_iterations,
            "KLD Loss Train": kld_l/total_iterations,
            "Ground Truth": rollout_metrics["Ground Truth"],
            "Imagination": rollout_metrics["Imagination"],
            #"Reconstruction Loss Test": loss_metrics["reconstruction_logprob"],
            "KLD Loss Test": loss_metrics["kl_loss"],
            "Logic Loss Test": loss_metrics["logic_loss"],
            "LTN Predictions": ltn_predictions["LTN Reconstruction"],
            "LTN Ground Truth": ltn_predictions["Ground Truth"]
        }
        wandb.log(metrics)
        #wandb.log({"Reconstruction Loss": recon_loss.item()})
        #wandb.log({"KLD Loss": kld_loss.item()})
        print(f"Epoch {epoch}: recon_loss={recon_loss.item():.2f}, kld_loss={kld_loss.item():.2f}, Logic loss:{logic_l}")
        logic_weight = logic_weight*logic_decay_rate
    
    wandb.finish()
    #save_model(encoder, epochs, "encoder", model_save_path)
    logic_loss_object.ltn_models.save_all_models(model_save_path)
    save_model(decoder, epochs, "decoder", model_save_path)
    save_model(rssm, epochs, "rssm", model_save_path)
    



