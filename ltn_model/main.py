from ltn_qm.ltn_qm import LTNRules
from dataset.dataset import get_dataset
import torch
from torch.utils.data import DataLoader
import wandb

def getreconstruction(dataloader_test, Front, Right, Up, dec, rot_plus, rot_minus):
    print("showreconstruction")
    initial_state_img_t, action_t, next_state_img_t = next(iter(dataloader_test))
    reconstructed = None
    
    with torch.no_grad():
        front = Front(initial_state_img_t.to("cuda")) #.view(initial_state_img.shape[0], -1)
        right = Right(initial_state_img_t.to("cuda")) #.view(initial_state_img.shape[0], -1)
        up = Up(initial_state_img_t.to("cuda")) #.view(initial_state_img.shape[0], -1)

        if action_t[0].item() == 0:
            decoded = dec(front[0].unsqueeze(0), rot_plus(right[0].unsqueeze(0)), front[0].unsqueeze(0)) 
            reconstructed = torch.nn.functional.sigmoid(decoded[0])
        elif action_t[0].item() == 1:
            decoded = dec(up[0].unsqueeze(0), rot_minus(right[0].unsqueeze(0)), up[0].unsqueeze(0))  
            reconstructed = torch.nn.functional.sigmoid(decoded[0])
        elif action_t[0].item() == 2:
            decoded = dec(right[0].unsqueeze(0), right[0].unsqueeze(0), rot_plus(up[0].unsqueeze(0))) 
            reconstructed = torch.nn.functional.sigmoid(decoded[0])
        elif action_t[0].item() == 3:
            decoded = dec(front[0].unsqueeze(0), rot_minus(up[0].unsqueeze(0)), front[0].unsqueeze(0))  
            reconstructed = torch.nn.functional.sigmoid(decoded[0])
        elif action_t[0].item() == 4:
            decoded = dec(rot_plus(front[0].unsqueeze(0)), rot_plus(up[0].unsqueeze(0)), up[0].unsqueeze(0))  
            reconstructed = torch.nn.functional.sigmoid(decoded[0])
        elif action_t[0].item() == 5:
            decoded = dec(rot_minus(front[0].unsqueeze(0)), right[0].unsqueeze(0), rot_minus(right[0].unsqueeze(0))) 
            reconstructed = torch.nn.functional.sigmoid(decoded[0])
    
    return { "GT_image_1": initial_state_img_t[0], "GT_image_2": next_state_img_t[0], "Reconstruction": reconstructed}


def train(ltn_obj, optimizer, dataloader_train, dataloader_test, epochs):
    for epoch in range(epochs):
        train_loss = 0
        train_sat = 0

        for init_image_, action_, next_image_ in dataloader_train:
            sat_val = ltn_obj.compute_sat(init_image_.float().to("cuda"), action_.float().to("cuda"), next_image_.float().to("cuda"))
            
            optimizer.zero_grad()
            loss = 1.-sat_val
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_sat += sat_val

        train_loss = train_loss/len(dataloader_train)
        train_sat = train_sat/len(dataloader_train)
        log_dict = getreconstruction(dataloader_test, ltn_obj.logic_models.front, ltn_obj.logic_models.right, ltn_obj.logic_models.up, ltn_obj.logic_models.dec, ltn_obj.logic_models.rot_plus, ltn_obj.logic_models.rot_minus)
        wandb.log({ "gt_image_1": wandb.Image(log_dict["GT_image_1"]), "gt_image_2": wandb.Image(log_dict["GT_image_2"]), "reconstruction": wandb.Image(log_dict["Reconstruction"]), "Epoch": epoch, "Loss": train_loss, "Sat": train_sat })
        print(f"Epoch: {epoch} | Train loss: {train_loss} | Train Sat:{train_sat}")

def main(dataset_train_path, dataset_test_path, login_key, batch_size=32, lr=0.00001, epochs=1):
    wandb.login(key=login_key)
    wandb.init(project="QM_LTN")

    LTNObject = LTNRules()
    optimizer = torch.optim.Adam(LTNObject.logic_models.get_params(), lr=lr)
    
    dataset_train = get_dataset(dataset_path=dataset_train_path)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    
    dataset_test = get_dataset(dataset_path=dataset_test_path)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    train(ltn_obj=LTNObject, optimizer=optimizer, dataloader_train=dataloader_train, dataloader_test=dataloader_test, epochs=epochs)
    wandb.finish()


main("../../dataset/cube_traj_dataset_train.npz", "../../dataset/cube_traj_dataset_test.npz", "9344322800342d92a14b2a59c0a58278aa02ae93")

    

