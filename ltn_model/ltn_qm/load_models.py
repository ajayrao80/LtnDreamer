import torch
from modules.orientation import RotQMinus
from modules.orientation import RotQPlus
from modules.face import Face
from modules.decoder import Decoder
from modules.digit import DigitClassifier

class LoadModels:
    def __init__(self, front_weights_path, right_weights_path, up_weights_path, dec_weights_path, rot_plus_weights_path, rot_minus_weights_path, digits_weights_path):
        self.front = Face().to(self.device) 
        self.right = Face().to(self.device)
        self.up = Face().to(self.device) 
        self.dec = Decoder().to(self.device) 
        self.rot_plus = RotQPlus().to(self.device) 
        self.rot_minus = RotQMinus().to(self.device)
        self.digits = DigitClassifier().to(self.device)

        self.front = self.load_model(front_weights_path, self.front)
        self.right = self.load_model(right_weights_path, self.right)
        self.up = self.load_model(up_weights_path, self.up)
        self.dec = self.load_model(dec_weights_path, self.dec)
        self.rot_plus = self.load_model(rot_plus_weights_path, self.rot_plus)
        self.rot_minus = self.load_model(rot_minus_weights_path, self.rot_minus)
        self.digits = self.load_model(digits_weights_path, self.digits)

        return self.front, self.right, self.up, self.dec, self.rot_plus, self.rot_minus, self.digits

    def load_model(self, weights_path, model_object):
        return model_object.load_state_dict(torch.load(weights_path, weights_only=True))
    


