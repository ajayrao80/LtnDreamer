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
from PIL import Image, ImageDraw
import numpy as np
from collections import defaultdict
import cv2
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from scipy.spatial.distance import cosine

def crop_to_content(pil_img):
    img_array = np.array(pil_img.convert('L'))
    
    _, thresh = cv2.threshold(img_array, 10, 255, cv2.THRESH_BINARY)
    
    # Find bounding box of the white content
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return pil_img # Return original if empty
        
    x, y, w, h = cv2.boundingRect(coords)
    
    # Crop the original PIL image
    return pil_img.crop((x, y, x+w, y+h))

def get_similarity_score(img1_pil, img2_pil):
    # Define the transformation pipeline
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img1_cropped = crop_to_content(img1_pil)
    img2_cropped = crop_to_content(img2_pil)
    
    img1_rgb = img1_cropped.convert("RGB")
    img2_rgb = img2_cropped.convert("RGB")

    t1 = preprocess(img1_rgb).unsqueeze(0)
    t2 = preprocess(img2_rgb).unsqueeze(0)
    
    return 1-cosine(t1.flatten(), t2.flatten()) 

def get_image_parts(img):
  result = []
  image = (img * 255).astype(np.uint8)
  img_ = Image.fromarray(image)

  # Right face
  points_right = [
      (65, 69),   # top-left
      (95, 50),  # top-right
      (98, 90), # bottom-right
      (70, 110)   # bottom-left
  ] 

  # front face 
  points_front = [
      (30, 55),   # top-left
      (66, 70),  # top-right
      (66, 110), # bottom-right
      (30, 97)   # bottom-left
  ] 

  # up
  points_up = [
      (59, 31),   # top-left
      (96, 50),  # top-right
      (66, 71), # bottom-right
      (29, 55)   # bottom-left
  ] 

  mask_f = Image.new("L", img_.size, 0)
  draw = ImageDraw.Draw(mask_f)
  draw.polygon(points_front, fill=255)
  result_front = Image.new("RGB", img_.size)
  result_front.paste(img_, (0, 0), mask_f)
  result.append(result_front)

  mask_r = Image.new("L", img_.size, 0)
  draw = ImageDraw.Draw(mask_r)
  draw.polygon(points_right, fill=255)
  result_right = Image.new("RGB", img_.size)
  result_right.paste(img_, (0, 0), mask_r)
  result.append(result_right)

  mask_u = Image.new("L", img_.size, 0)
  draw = ImageDraw.Draw(mask_u)
  draw.polygon(points_up, fill=255)
  result_up = Image.new("RGB", img_.size)
  result_up.paste(img_, (0, 0), mask_u)
  result.append(result_up)

  return result

def get_next_state(states, actions_ids):
    tu = torch.tensor([
    [0., 0., 0., 0., 0., 1],
    [0., 1., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 1., 0., 0.]
    ])

    td = torch.tensor([
        [0., 0., 1., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 0.],
        [1., 0., 0., 0., 0., 0.]
    ])

    tl = torch.tensor([
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0.],
        [1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1.]
    ])

    tr = torch.tensor([
        [0., 0., 0., 0., 1., 0.],
        [1., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1.]
    ])

    tur = torch.tensor([
        [1., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0., 0.]
    ])

    tul = torch.tensor([
        [1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0.]
    ])

    actions = torch.stack([tu.T, td.T, tl.T, tr.T, tur.T, tul.T], dim=0)

    next_states = torch.matmul(states.unsqueeze(1), actions[actions_ids]).squeeze(1)
    return next_states

def encoder_ltn(obs, logic_models):
    front = logic_models.front(obs)
    right = logic_models.right(obs)
    up = logic_models.up(obs)
    return torch.cat([front, right, up], dim=1).flatten(1)

def get_episode_metrics(episode, encoder, rssm, decoder, logic_loss_object, upscale_network, vanilla_dreamer=True):
    logic_loss_object.ltn_models.front.eval()
    logic_loss_object.ltn_models.right.eval()
    logic_loss_object.ltn_models.up.eval() #encoder.eval()
    decoder.eval()
    rssm.eval()
    B = 1
    
    with torch.no_grad():
        obs = torch.tensor(episode["observation"]).unsqueeze(0)
        actions = torch.tensor(episode["actions"]).unsqueeze(0)
        T = obs.shape[1]

        step_errors = { "mse_step_error": torch.zeros(T-1), "logic_step_error": torch.zeros(T-1), "similarity_score": torch.zeros(T-1)}

        deter_dim = rssm.rnn.hidden_size
        stoch_dim = rssm.fc_prior.out_features // 2

        device = obs.device

        deter = torch.zeros(B, deter_dim, device=device)
        stoch = torch.zeros(B, stoch_dim, device=device)

        if vanilla_dreamer:
            embed = encoder(obs[:, 0]) 
        else:
            embed = encoder(obs[:, 0], logic_loss_object.ltn_models) 

        state = torch.tensor([0., 1., 2., 3., 4., 5.]).unsqueeze(0)
        
        for t in range(1, T):
            prior_stoch, prior_mean, prior_std, post_stoch, post_mean, post_std, deter = rssm(
                stoch, deter, actions[:, t-1], embed)
            
            f, r, u = upscale_network(post_stoch)
            if vanilla_dreamer:
                mean = decoder(post_stoch) #decoder(post_stoch) 
            else:
                mean = decoder(f, r, u) 
            
            actions_batch = actions[:, t-1].max(dim=1, keepdim=True).values #.squeeze(1)
            logic_loss = logic_loss_object.compute_logic_loss(obs[:, t-1], actions_batch, mean) 
            logic_loss += logic_loss_object.set_encodings_equal(obs[:, t-1], actions_batch, f, r, u)
            mse_loss = torch.nn.functional.mse_loss(obs[:, t], mean)
            step_errors["mse_step_error"][t-1] = mse_loss
            step_errors["logic_step_error"][t-1] = logic_loss
            
            # similarity score for cropped faces
            action = int(actions[:, t-1].max().item())
            state = get_next_state(state, action) 
            similarity_score = 1.
            f_gt, r_gt, u_gt = get_image_parts(obs[:, t][0].numpy().transpose(1, 2, 0))
            f_pred, r_pred, u_pred = get_image_parts(mean[0].numpy().transpose(1, 2, 0))
            f_sim = get_similarity_score(f_gt, f_pred).item()
            r_sim = get_similarity_score(r_gt, r_pred).item()
            u_sim = get_similarity_score(u_gt, u_pred).item()
            compared_faces = 0
            if state[0][0] <= 2:
                similarity_score += f_sim
                compared_faces += 1
            if state[0][1] <= 2:
                similarity_score += r_sim
                compared_faces += 1
            if state[0][2] <= 2:
                compared_faces += 1
                similarity_score += u_sim

            #if compared_faces > 0:
            #    step_errors["similarity_score"][t-1] = (f_sim + r_sim + u_sim) / (compared_faces)
            #else:
            #    step_errors["similarity_score"][t-1] = (f_sim + r_sim + u_sim) / 3.0
            step_errors["similarity_score"][t-1] = (f_sim + r_sim + u_sim) / 3.0
            
            stoch = post_stoch
            embed=None
            
    return step_errors

def main(dataset_test_path, vanilla_model_path, logic_models_path, vanilla_model=True):
    dataset_test = np.load(dataset_test_path)
    obs = dataset_test["observation"]
    actions = dataset_test["action"]
    obs_shape = (3, 128, 128)
    action_dim = 7
    T = 5
    stoch_dim = 200
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    upscale_network = UpscaleNetwork(stoch_dim).to(device)
    upscale_network.load_state_dict(torch.load(f"{logic_models_path}/world_model_upscale_network_1000", weights_only=True, map_location=torch.device(device)))
    logic_loss_object = LogicLoss(logic_models_path, model_name_digits=None, train_all=False)

    if vanilla_model:
        embed_dim = 200
        deter_dim = 400
        encoder = Encoder(obs_shape, embed_dim).to(device)
        encoder.load_state_dict(torch.load(f"{vanilla_model_path}/world_model_encoder_1000", weights_only=True, map_location=torch.device(device)))
        decoder = Decoder(embed_dim, obs_shape).to(device)
        decoder.load_state_dict(torch.load(f"{vanilla_model_path}/world_model_decoder_1000", weights_only=True, map_location=torch.device(device)))
        rssm = RSSM(action_dim, stoch_dim, deter_dim, embed_dim).to(device)
        rssm.load_state_dict(torch.load(f"{vanilla_model_path}/world_model_rssm_1000", weights_only=True, map_location=torch.device(device)))
    else:
        embed_dim = 3*128*14*14
        deter_dim = 200
        rssm = RSSM(action_dim, stoch_dim, deter_dim, embed_dim).to(device)
        rssm.load_state_dict(torch.load(f"{logic_models_path}/world_model_rssm_1000", weights_only=True, map_location=torch.device(device)))
    
    num_episodes = int(len(obs)/T)
    mse_errors = torch.zeros(int(len(obs)/T), T-1, device=device)
    logic_errors = torch.zeros(int(len(obs)/T), T-1, device=device)
    similarity_scores = torch.zeros(int(len(obs)/T), T-1, device=device)
    count = 0
    for i in range(0, len(obs), T):
        sample_obs = [obs[i] for i in range(i, i+T)]
        sample_actions = [actions[i] for i in range(i, i+T)]
        episode = {"observation": sample_obs, "actions": sample_actions}
        if vanilla_model:
            error = get_episode_metrics(episode, encoder, rssm, decoder, logic_loss_object, upscale_network) 
        else:
            error = get_episode_metrics(episode, encoder_ltn, rssm, logic_loss_object.ltn_models.dec, logic_loss_object, upscale_network, vanilla_dreamer=False)     
        mse_errors[count, :] = error["mse_step_error"]
        logic_errors[count, :] = error["logic_step_error"]
        similarity_scores[count, :] = error["similarity_score"]
        count += 1
        print(f"{count}/{num_episodes}")
    
    if vanilla_model:
        np.save(f"{vanilla_model_path}/mse_errors_vanilla.npy", mse_errors.cpu().numpy())
        np.save(f"{vanilla_model_path}/logic_errors_vanilla.npy", logic_errors.cpu().numpy())
        np.save(f"{vanilla_model_path}/similarity_scores_vanilla.npy", similarity_scores.cpu().numpy())
    else:
        np.save(f"{logic_models_path}/mse_errors_ltn_dreamer.npy", mse_errors.cpu().numpy())
        np.save(f"{logic_models_path}/logic_errors_ltn_dreamer.npy", logic_errors.cpu().numpy())
        np.save(f"{logic_models_path}/similarity_scores_ltn_dreamer.npy", similarity_scores.cpu().numpy())

    

















