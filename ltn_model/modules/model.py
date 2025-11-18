import torch
import torch.nn as nn
from modules.orientation import RotQMinus
from modules.orientation import RotQPlus
from modules.face import Face
from modules.decoder import Decoder
from modules.digit import DigitClassifier
from itertools import chain

class LTNModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.front = Face().to(self.device)
        self.right = Face().to(self.device)
        self.up = Face().to(self.device)
        self.dec = Decoder().to(self.device)
        self.rot_plus = RotQPlus().to(self.device)
        self.rot_minus = RotQMinus().to(self.device)
        self.digits = DigitClassifier().to(self.device) #[DigitClassifier().to(self.device) for _ in range(10)]
    
    def get_params(self):
        #params = chain(
        #    *[model.parameters() for model in [self.front, self.right, self.up, self.dec, self.rot_plus, self.rot_minus, self.digits]]
        #)

        params = list(self.front.parameters()) + list(self.right.parameters()) + list(self.up.parameters()) + list(self.dec.parameters()) + list(self.rot_plus.parameters()) + list(self.rot_minus.parameters()) + list(self.digits.parameters())

        return params


        
