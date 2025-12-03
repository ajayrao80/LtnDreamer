import argparse
from dreamer.dataset.dataset import Dataset
from dreamer.modules.encoder import Encoder
from dreamer.modules.decoder import Decoder
from dreamer.modules.model import RSSM
from dreamer.modules.memory import DynamicsModel
from dreamer.modules.upscale import UpscaleNetwork
import torch
from dreamer.utils.utils import log
import wandb
from dreamer.utils.utils import save_model
from ltn_model.ltn_qm.logic_loss import LogicLoss

def eval_loss(dataset, dynamics_model, decoder, logic_loss_object, T=5, batch_size=32, obs_shape=(3, 128, 128), device=None):
    logic_loss_object.ltn_models.front.eval()
    logic_loss_object.ltn_models.right.eval()
    logic_loss_object.ltn_models.up.eval() #encoder.eval()
    decoder.eval()
    dynamics_model.eval()
    sample = dataset.sample(batch_size, T)
    obs = sample.observation
    actions = sample.action.max(dim=2, keepdim=True).values
    with torch.no_grad():
        total_loss = 0.
        state = torch.zeros(batch_size, obs_shape[0], obs_shape[1], obs_shape[2]).to(device)
        for t in range(1, T):
            action_batch = actions[:, t-1].max(dim=1, keepdim=True).values #.squeeze(1)
            state = dynamics_model(state, obs[:, t-1], action_batch) #[:, t-1])    
            reconstructed_image = decoder(logic_loss_object.ltn_models.front(state), logic_loss_object.ltn_models.right(state), logic_loss_object.ltn_models.up(state)) 

            ltn_loss = logic_loss_object.compute_logic_loss(obs[:, t-1], action_batch, obs[:, t]) 
            logic_loss = logic_loss_object.compute_logic_loss(obs[:, t-1], action_batch, reconstructed_image) 
            total_loss += ltn_loss + logic_loss 
            
        total_loss = total_loss/T

        metrics = {
            'loss': total_loss.item()
        }

    return metrics

def eval_rollout(dataset, dynamics_model, decoder, logic_loss_object, T=5, obs_shape=(3, 128, 128)):
    with torch.no_grad():
        sample = dataset.sample(2, T)
        initial_obs = sample.observation
        action_seq = sample.action.max(dim=2, keepdim=True).values
        
        B = initial_obs.size(0)
        device = initial_obs.device

        ground_truth_images = [wandb.Image(sample.observation[0, 0])]
        reconstructed_images = []

        state = torch.zeros(B, obs_shape[0], obs_shape[1], obs_shape[2]).to(device)
        
        for t in range(1, T):
            actions = action_seq[:, t-1].max(dim=1, keepdim=True).values #.squeeze(1)
            state = dynamics_model(state, initial_obs[:, t-1], actions) #[0, t-1].unsqueeze(1))    
            reconstructed_image = decoder(logic_loss_object.ltn_models.front(state), logic_loss_object.ltn_models.right(state), logic_loss_object.ltn_models.up(state)) 
            
            ground_truth_images.append(wandb.Image(sample.observation[0, t-1]))
            reconstructed_images.append(wandb.Image(reconstructed_image[0]))

        roll_outs = {
            "Ground Truth": ground_truth_images,
            "Imagination": reconstructed_images
        }

        return roll_outs

def get_ltn_predictions(dataset, logic_loss_object, T=5):
    with torch.no_grad():
        sample = dataset.sample(1, T)
        initial_obs = sample.observation[0, 0].unsqueeze(0)
        action = sample.action[0, 0].unsqueeze(0)
        action = action.max(dim=1, keepdim=True).values #.squeeze(1)

        ltn_reconstruction_pred = logic_loss_object.get_ltn_predictions(initial_obs, action)
        return {"Base image (ground truth)":sample.observation[0, 0], "LTN Reconstruction": wandb.Image(ltn_reconstruction_pred[0]), "Ground Truth": wandb.Image(sample.observation[0, 1])}

def main(lr, epochs, embed_dim, dataset_train_path, dataset_test_path, login_key, model_save_path, logic_models_path=None, project_name="vanilla_world_model", train_all=True, batch_size=32):
    obs_shape = (3, 128, 128)
    action_dim = 7
    embed_dim = embed_dim

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    dataset_object = Dataset(obs_shape, action_dim, device, dataset_train_path, dataset_test_path)
    dataset_train = dataset_object.get_dataset_train()
    dataset_test = dataset_object.get_dataset_test()

    action_dim = 1 # dataset loader wants one-hot encoding and dynamics model wants index of action. Just a hack to make both happy.

    B, T = batch_size, 5
    total_iterations = int(dataset_train.observation.shape[0] / B)
    epochs = epochs

    logic_loss_object = LogicLoss(logic_models_path, model_name_digits=None, train_all=train_all)
    dynamics_model = DynamicsModel(embed_dim, logic_model=logic_loss_object.ltn_models if logic_loss_object is not None else None, obs_shape=obs_shape, action_dim=action_dim).to(device)
    decoder = logic_loss_object.ltn_models.dec

    optim_model = torch.optim.Adam(list(dynamics_model.parameters()) + logic_loss_object.get_logic_parameters(), lr=lr) 

    wandb.login(key=login_key)
    wandb.init(project=project_name)

    for epoch in range(epochs): 
        l = 0.

        for iteration in range(total_iterations):
            logic_loss_total = 0.

            sample = dataset_train.sample(B, T)
            obs = sample.observation
            actions = sample.action.max(dim=2, keepdim=True).values
            state = torch.zeros(B, obs_shape[0], obs_shape[1], obs_shape[2]).to(device)
            for t in range(1, T):
                actions_batch = actions[:, t-1].max(dim=1, keepdim=True).values #.squeeze(1)
                state = dynamics_model(state, obs[:, t-1], actions_batch)     
                recon_mean = decoder(logic_loss_object.ltn_models.front(state), logic_loss_object.ltn_models.right(state), logic_loss_object.ltn_models.up(state)) 

                ltn_loss = logic_loss_object.compute_logic_loss(obs[:, t-1], actions_batch, obs[:, t]) if train_all else 0.
                logic_loss = logic_loss_object.compute_logic_loss(obs[:, t-1], actions_batch, recon_mean) if logic_models_path is not None else 0.
                
                logic_loss_total += ltn_loss + logic_loss 
            
            loss = logic_loss_total
            optim_model.zero_grad()
            loss.backward()
            optim_model.step()

            l += loss.item()
            
        rollout_metrics = eval_rollout(dataset_test, dynamics_model, decoder, logic_loss_object)
        loss_metrics = eval_loss(dataset_test, dynamics_model, decoder, logic_loss_object, device=device)
        ltn_predictions = get_ltn_predictions(dataset_test, logic_loss_object)

        metrics = {
            "Epoch": epoch,
            "Loss": l/total_iterations,
            "Ground Truth": rollout_metrics["Ground Truth"],
            "Imagination": rollout_metrics["Imagination"],
            "Logic Loss Test": loss_metrics["logic_loss"],
            "LTN Predictions": ltn_predictions["LTN Reconstruction"],
            "LTN Ground Truth": ltn_predictions["Ground Truth"]
        }
        wandb.log(metrics)
        #wandb.log({"Reconstruction Loss": recon_loss.item()})
        #wandb.log({"KLD Loss": kld_loss.item()})
        print(f"Epoch {epoch}: loss:{l}")
    
    wandb.finish()
    logic_loss_object.ltn_models.save_all_models(model_save_path)
    save_model(dynamics_model, epochs, "dynamics", model_save_path)
    



