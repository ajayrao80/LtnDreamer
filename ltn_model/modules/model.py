import torch
import torch.nn as nn
from ltn_model.modules.orientation import RotQMinus
from ltn_model.modules.orientation import RotQPlus
from ltn_model.modules.face import Face
from ltn_model.modules.decoder import Decoder
from ltn_model.modules.digit import DigitClassifier
from itertools import chain
from ltn_model.utils.utils import save_model

class LTNModel:
    def __init__(self, front_=None, right_=None, up_=None, dec_=None, rot_plus_=None, rot_minus_=None, digits_=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.front = Face().to(self.device) if front_ is None else front_
        self.right = Face().to(self.device) if right_ is None else right_
        self.up = Face().to(self.device) if up_ is None else up_
        self.dec = Decoder().to(self.device) if dec_ is None else dec_
        self.rot_plus = RotQPlus().to(self.device) if rot_plus_ is None else rot_plus_
        self.rot_minus = RotQMinus().to(self.device) if rot_minus_ is None else rot_minus_
        #self.digits = DigitClassifier().to(self.device) if digits_ is None else digits_ #[DigitClassifier().to(self.device) for _ in range(10)]
    
    def get_params(self):
        #params = chain(
        #    *[model.parameters() for model in [self.front, self.right, self.up, self.dec, self.rot_plus, self.rot_minus, self.digits]]
        #)
        params = list(self.front.parameters()) + list(self.right.parameters()) + list(self.up.parameters()) + list(self.dec.parameters()) + list(self.rot_plus.parameters()) + list(self.rot_minus.parameters()) #+ list(self.digits.parameters())
        return params
    
    def save_all_models(self, save_path):
        save_model(self.front, save_path, "front")
        save_model(self.right, save_path, "right")
        save_model(self.up, save_path, "up")
        save_model(self.dec, save_path, "dec")
        save_model(self.rot_plus, save_path, "rot_plus")
        save_model(self.rot_minus, save_path, "rot_minus")
        #save_model(self.digits, save_path, "digits")
    
    def load_model(self, path, model):
        model.load_state_dict(torch.load(path, weights_only=True))
        return model
    
    def load_all_models(self, model_path, front, right, up, dec, rot_plus, rot_minus, digits=None):
        self.front = self.load_model(f"{model_path}/{front}", self.front)
        self.right = self.load_model(f"{model_path}/{right}", self.right)
        self.up = self.load_model(f"{model_path}/{up}", self.up)
        self.dec = self.load_model(f"{model_path}/{dec}", self.dec)
        self.rot_plus = self.load_model(f"{model_path}/{rot_plus}", self.rot_plus)
        self.rot_minus = self.load_model(f"{model_path}/{rot_minus}", self.rot_minus)
        #self.digits = self.load_model(f"{model_path}/{digits}", self.digits) if digits is not None else self.digits
        


        


