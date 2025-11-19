import torch
import torch.nn as nn
from modules.orientation import RotQMinus
from modules.orientation import RotQPlus
from modules.face import Face
from modules.decoder import Decoder
from modules.digit import DigitClassifier
from itertools import chain
from utils.utils import save_model

class LTNModel:
    def __init__(self, front_=None, right_=None, up_=None, dec_=None, rot_plus_=None, rot_minus_=None, digits_=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.front = Face().to(self.device) if front_ is None else front_
        self.right = Face().to(self.device) if right_ is None else right_
        self.up = Face().to(self.device) if up_ is None else up_
        self.dec = Decoder().to(self.device) if dec_ is None else dec_
        self.rot_plus = RotQPlus().to(self.device) if rot_plus_ is None else rot_plus_
        self.rot_minus = RotQMinus().to(self.device) if rot_minus_ is None else rot_minus_
        self.digits = DigitClassifier().to(self.device) if digits_ is None else digits_ #[DigitClassifier().to(self.device) for _ in range(10)]
    
    def get_params(self):
        #params = chain(
        #    *[model.parameters() for model in [self.front, self.right, self.up, self.dec, self.rot_plus, self.rot_minus, self.digits]]
        #)
        params = list(self.front.parameters()) + list(self.right.parameters()) + list(self.up.parameters()) + list(self.dec.parameters()) + list(self.rot_plus.parameters()) + list(self.rot_minus.parameters()) + list(self.digits.parameters())
        return params
    
    def save_all_models(self, save_path):
        save_model(self.front, save_path, "front")
        save_model(self.right, save_path, "right")
        save_model(self.up, save_path, "up")
        save_model(self.dec, save_path, "dec")
        save_model(self.rot_plus, save_path, "rot_plus")
        save_model(self.rot_minus, save_path, "rot_minus")
        save_model(self.digits, save_path, "digits")


        


