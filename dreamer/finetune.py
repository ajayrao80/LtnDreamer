import argparse
from dreamer.dataset.dataset import Dataset
from dreamer.modules.encoder import Encoder
from dreamer.modules.decoder import Decoder
from dreamer.modules.model import RSSM
import torch
from dreamer.utils.utils import log
import wandb
from dreamer.utils.utils import save_model
from ltn_model.ltn_qm.logic_loss import LogicLoss

def eval_loss(dataset, encoder, rssm, decoder, logic_loss_object, T=5, batch_size=32):
    encoder.eval()
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
            embed = encoder(obs[:, t-1])
            prior_stoch, prior_mean, prior_std, post_stoch, post_mean, post_std, deter = rssm(
                stoch, deter, actions[:, t-1], embed)
            
            mean = decoder(post_stoch)
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

            reconstructed_image = decoder(post_stoch)

            ground_truth_images.append(wandb.Image(sample.observation[0, t]))
            reconstructed_images.append(wandb.Image(reconstructed_image[0]))

            stoch_state = post_stoch
            embed=None

        roll_outs = {
            "Ground Truth": ground_truth_images,
            "Imagination": reconstructed_images
        }

        return roll_outs

def main(lr, epochs, embed_dim, stoch_dim, deter_dim, dataset_train_path, dataset_test_path, beta, login_key, model_save_path, dreamer_load_path=None, logic_models_path=None, free_nats=3.0, project_name="vanilla_world_model", logic_weight=15000.0, logic_decay_rate=0.99, batch_size=32):
    obs_shape = (3, 128, 128)
    action_dim = 7
    embed_dim = embed_dim
    stoch_dim = stoch_dim
    deter_dim = deter_dim

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    encoder = Encoder(obs_shape, embed_dim).to(device)
    decoder = Decoder(embed_dim, obs_shape).to(device)
    rssm = RSSM(action_dim, stoch_dim, deter_dim, embed_dim).to(device)
    
    if dreamer_load_path is not None:
        encoder.load_state_dict(torch.load(f"{dreamer_load_path}/world_model_encoder_1000", weights_only=True))
        decoder.load_state_dict(torch.load(f"{dreamer_load_path}/world_model_decoder_1000", weights_only=True))
        rssm.load_state_dict(torch.load(f"{dreamer_load_path}/world_model_rssm_1000", weights_only=True))

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
        logic_loss_object = LogicLoss(logic_models_path, model_name_digits=None)
    
    optim_model = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(rssm.parameters()), lr=lr)

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

            for t in range(1, T):
                sample = dataset_train.sample(B, T)
                obs = sample.observation
                actions = sample.action
                embed = encoder(obs[:, t-1])
                prior_stoch, prior_mean, prior_std, post_stoch, post_mean, post_std, deter = rssm(stoch, deter, actions[:, t-1], embed)
                recon_mean = decoder(post_stoch)
                # turn off reconstruction loss for finetuning with logic only ---------------
                #fixed_std = 1.0
                #dist = torch.distributions.Normal(recon_mean, fixed_std)
                #recon_log_prob = dist.log_prob(obs[:, t]).sum(dim=[1,2,3]).mean()
                #recon_loss += -recon_log_prob
                # ---------------------------------------------------------------------------
                
                actions_batch = actions[:, t-1].max(dim=1, keepdim=True).values.squeeze(1)
                logic_loss = logic_loss_object.compute_logic_loss(obs[:, t-1], actions_batch, recon_mean) if logic_models_path is not None else 0.

                #ltn_loss = logic_loss_object.compute_logic_loss(obs[:, t-1], actions_batch, obs[:, t]) if train_all else 0.
                
                logic_loss_total += logic_loss #+ ltn_loss
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
            logic_loss_total = logic_loss_total*logic_weight
            kld_loss = (kld_loss * beta)
            loss =  kld_loss + logic_loss_total # + recon_loss
            optim_model.zero_grad()
            loss.backward()
            optim_model.step()

            l += loss.item()
            logic_l += logic_loss_total.item()
            #rl += recon_loss.item()
            kld_l += kld_loss.item()
        
        rollout_metrics = eval_rollout(dataset_test, encoder, rssm, decoder)
        loss_metrics = eval_loss(dataset_test, encoder, rssm, decoder, logic_loss_object)
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
            "Logic Loss Test": loss_metrics["logic_loss"]
        }
        wandb.log(metrics)
        #wandb.log({"Reconstruction Loss": recon_loss.item()})
        #wandb.log({"KLD Loss": kld_loss.item()})
        print(f"Epoch {epoch}: recon_loss={recon_loss.item():.2f}, kld_loss={kld_loss.item():.2f}, Logic loss:{logic_l}")
        #logic_weight = logic_weight*logic_decay_rate
    
    wandb.finish()
    save_model(encoder, epochs, "encoder", model_save_path)
    save_model(decoder, epochs, "decoder", model_save_path)
    save_model(rssm, epochs, "rssm", model_save_path)
    



