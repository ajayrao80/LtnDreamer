from ltn_qm.ltn_qm import LTNRules
from dataset.dataset import get_dataset
import torch
from torch.utils.data import DataLoader
import wandb

def getreconstruction(dataloader_test, Front, Right, Up, dec, rot_plus, rot_minus, digits):
    print("showreconstruction")
    initial_state_img_t, numbers_init_t, action_t, next_state_img_t, numbers_next_t = next(iter(dataloader_test))
    reconstructed = None
    
    with torch.no_grad():
        front = Front(initial_state_img_t.to("cuda")) #.view(initial_state_img.shape[0], -1)
        right = Right(initial_state_img_t.to("cuda")) #.view(initial_state_img.shape[0], -1)
        up = Up(initial_state_img_t.to("cuda")) #.view(initial_state_img.shape[0], -1)

        #front_digit_classification = torch.argmax(torch.tensor([P(front[0].unsqueeze(0)).item() for P in digits])).item()
        #right_digit_classification = torch.argmax(torch.tensor([P(right[0].unsqueeze(0)).item() for P in digits])).item()
        #up_digit_classification = torch.argmax(torch.tensor([P(up[0].unsqueeze(0)).item() for P in digits])).item()

        front_digit_classification = digits(front[0].unsqueeze(0)).argmax().item()
        right_digit_classification = digits(right[0].unsqueeze(0)).argmax().item()
        up_digit_classification = digits(up[0].unsqueeze(0)).argmax().item()

        if action_t[0].item() == 0:
            decoded = dec(front[0].unsqueeze(0), rot_plus(right[0].unsqueeze(0)), front[0].unsqueeze(0)) 
            #reconstructed = torch.nn.functional.sigmoid(decoded[0])
        elif action_t[0].item() == 1:
            decoded = dec(up[0].unsqueeze(0), rot_minus(right[0].unsqueeze(0)), up[0].unsqueeze(0))  
            #reconstructed = torch.nn.functional.sigmoid(decoded[0])
        elif action_t[0].item() == 2:
            decoded = dec(right[0].unsqueeze(0), right[0].unsqueeze(0), rot_plus(up[0].unsqueeze(0))) 
            #reconstructed = torch.nn.functional.sigmoid(decoded[0])
        elif action_t[0].item() == 3:
            decoded = dec(front[0].unsqueeze(0), rot_minus(up[0].unsqueeze(0)), front[0].unsqueeze(0))  
            #reconstructed = torch.nn.functional.sigmoid(decoded[0])
        elif action_t[0].item() == 4:
            decoded = dec(rot_plus(front[0].unsqueeze(0)), rot_plus(up[0].unsqueeze(0)), up[0].unsqueeze(0))  
            #reconstructed = torch.nn.functional.sigmoid(decoded[0])
        elif action_t[0].item() == 5:
            decoded = dec(rot_minus(front[0].unsqueeze(0)), right[0].unsqueeze(0), rot_minus(right[0].unsqueeze(0))) 
            #reconstructed = torch.nn.functional.sigmoid(decoded[0])
    
    return { "GT_image_1": initial_state_img_t[0], "GT_image_2": next_state_img_t[0], "Reconstruction": torch.nn.functional.tanh(decoded[0]).clamp(0, 1), "front_digit": front_digit_classification, "right_digit": right_digit_classification, "up_digit": up_digit_classification}

def log(ltn_obj, dataloader_test, epoch, train_loss=None, train_sat=None):
    log_dict = getreconstruction(dataloader_test, ltn_obj.logic_models.front, ltn_obj.logic_models.right, ltn_obj.logic_models.up, ltn_obj.logic_models.dec, ltn_obj.logic_models.rot_plus, ltn_obj.logic_models.rot_minus, ltn_obj.logic_models.digits)
    metrics = { "gt_image_1": wandb.Image(log_dict["GT_image_1"]), "gt_image_2": wandb.Image(log_dict["GT_image_2"]), "reconstruction": wandb.Image(log_dict["Reconstruction"]), 
                "front_digit": log_dict["front_digit"], "right_digit": log_dict["right_digit"], "up_digit": log_dict["up_digit"], "Epoch": epoch}
    if train_loss is not None:
        metrics["Loss"] = train_loss
        metrics["Sat"] = train_sat
    wandb.log(metrics)

def train(ltn_obj, optimizer, dataloader_train, dataloader_test, epochs, model_save_path, steps_log=20):
    for epoch in range(epochs):
        train_loss = 0
        train_sat = 0

        #i = 0
        #for init_image_, numbers_init_, action_, next_image_, numbers_next_ in dataloader_train:
        for i, (init_image_, numbers_init_, action_, next_image_, numbers_next_) in enumerate(dataloader_train):
            sat_val = ltn_obj.compute_sat(init_image_.float().to("cuda"), action_.float().to("cuda"), next_image_.float().to("cuda"), numbers_init_.float().to("cuda"), numbers_next_.float().to("cuda"))
            
            optimizer.zero_grad()
            loss = 1.-sat_val
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_sat += sat_val

            if i % steps_log == 0:
                log(ltn_obj=ltn_obj, dataloader_test=dataloader_test, epoch=epoch) #, train_loss=train_loss/((i+1)*steps_log), train_sat=train_sat/((i+1)*steps_log))

        train_loss = train_loss/len(dataloader_train)
        train_sat = train_sat/len(dataloader_train)
        log(ltn_obj=ltn_obj, dataloader_test=dataloader_test, epoch=epoch, train_loss=train_loss, train_sat=train_sat)
        print(f"Epoch: {epoch} | Train loss: {train_loss} | Train Sat:{train_sat}")
        ltn_obj.logic_models.save_all_models(model_save_path)

def main(dataset_train_path, dataset_test_path, model_save_path, login_key, batch_size=32, lr=0.00001, epochs=1):
    wandb.login(key=login_key)
    wandb.init(project="QM_LTN")

    LTNObject = LTNRules()
    optimizer = torch.optim.Adam(LTNObject.logic_models.get_params(), lr=lr)
    
    dataset_train = get_dataset(dataset_path=dataset_train_path)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    
    dataset_test = get_dataset(dataset_path=dataset_test_path)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    train(ltn_obj=LTNObject, optimizer=optimizer, dataloader_train=dataloader_train, dataloader_test=dataloader_test, epochs=epochs, model_save_path=model_save_path)
    wandb.finish()

    

 