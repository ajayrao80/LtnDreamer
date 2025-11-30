from PIL import Image
import numpy as np
import torch

action_idx_name = { 0:"tu", 1:"td", 2:"tl", 3:"tr", 4:"tur", 5:"tul" } # names of the actions
action_name_idx = { "tu":0, "td":1, "tl":2, "tr":3, "tur":4, "tul":5 } # name to action id mapping
inverse_actions = { "tu":"td", "td":"tu", "tl":"tr", "tr":"tl", "tur":"tul", "tul":"tur" } 
face_idx_name = { 0:"front", 1:"up", 2:"right", 3:"back", 4:"down", 5:"left" }
orientation_idx_name = {0:"orient_up", 1:"orient_right", 2:"orient_down", 3:"orient_left"}

def show(image):
    if hasattr(image, 'detach'):
        image = image.detach().cpu().numpy()

    if image.shape[0] == 3:  # CHW to HWC
        image = np.transpose(image, (1, 2, 0))

    image = (image * 255).astype(np.uint8)
    img_pil = Image.fromarray(image)
    
    display(img_pil)

def shear(img, shear_factor_x=0.3, shear_factor_y=0.3, a=0, b=0, c=1, d=1):
    width, height = img.size
    matrix = (a, shear_factor_x, c, shear_factor_y, b, d)
    sheared_img = img.transform(
        (width, height),
        Image.AFFINE,
        matrix,
        resample=Image.BICUBIC
    )

    return sheared_img

def crop(image, left = 50, top = 50, window_size=100):
    if hasattr(image, 'detach'):
        image = image.detach().cpu().numpy()

    if image.shape[0] == 3:  # CHW to HWC
        image = np.transpose(image, (1, 2, 0))

    image = (image * 255).astype(np.uint8)
    img_pil = Image.fromarray(image)

    right = left + window_size
    bottom = top + window_size

    window = (left, top, right, bottom)

    cropped_image = img_pil.crop(window)
    return cropped_image

def cosine_similarity(A, B):
    A_flat = A.view(A.size(0), -1)  
    B_flat = B.view(B.size(0), -1)

    #if torch.all(A == 0) or torch.all(B == 0):
    #    return 0.0
    
    dot_product = torch.sum(A_flat * B_flat, dim=1)
    
    norm_A = torch.norm(A_flat, dim=1)
    norm_B = torch.norm(B_flat, dim=1)
    
    similarity = ((dot_product / (norm_A * norm_B)) + 1.0)/2.0
    return torch.nn.functional.sigmoid(similarity)

def save_model(model, path, name):
    torch.save(model.state_dict(), f"{path}/{name}")