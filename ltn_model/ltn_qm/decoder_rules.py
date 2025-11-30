import ltn
from ltn_model.ltn_qm.connectives_quantifiers import (Forall, Sim)
import torch

class DecoderRules:
    def __init__(self, ltn_F_and_P):
        # ltn functions and predicates
        self.ltn_F_and_P = ltn_F_and_P  

    def DecoderRuleA0(self, init_image, next_image, action, p=3):
        ltn.diag(init_image, next_image, action)
        return Forall([init_image, next_image, action],
            #Sim(self.ltn_F_and_P.Dec(self.ltn_F_and_P.Front(init_image), self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(init_image)), self.ltn_F_and_P.Front(init_image)), next_image), p=p
            Sim(self.ltn_F_and_P.Dec(self.ltn_F_and_P.Front(init_image), self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Right(init_image), action), self.ltn_F_and_P.Front(init_image)), next_image), p=p 
        )

    def DecoderRuleA1(self,init_image, next_image, action, p=3):
        ltn.diag(init_image, next_image, action)
        return Forall([init_image, next_image, action],
            #Sim(self.ltn_F_and_P.Dec(self.ltn_F_and_P.Up(init_image), self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(init_image)), self.ltn_F_and_P.Up(init_image)), next_image), p=p
            Sim(self.ltn_F_and_P.Dec(self.ltn_F_and_P.Up(init_image), self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Right(init_image), action), self.ltn_F_and_P.Up(init_image)), next_image), p=p 
        )
    
    def DecoderRuleA2(self, init_image, next_image, action, p=3):
        ltn.diag(init_image, next_image, action)
        return Forall([init_image, next_image, action],
            #Sim(self.ltn_F_and_P.Dec(self.ltn_F_and_P.Right(init_image), self.ltn_F_and_P.Right(init_image), self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(init_image))), next_image), p=p
            Sim(self.ltn_F_and_P.Dec(self.ltn_F_and_P.Right(init_image), self.ltn_F_and_P.Right(init_image), self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Up(init_image), action)), next_image), p=p 
        )

    def DecoderRuleA3(self, init_image, next_image, action, p=3):
        ltn.diag(init_image, next_image, action)
        return Forall([init_image, next_image, action],
            #Sim(self.ltn_F_and_P.Dec(self.ltn_F_and_P.Front(init_image), self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(init_image)), self.ltn_F_and_P.Front(init_image)), next_image), p=p
            Sim(self.ltn_F_and_P.Dec(self.ltn_F_and_P.Front(init_image), self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Up(init_image), action), self.ltn_F_and_P.Front(init_image)), next_image), p=p 
        )

    def DecoderRuleA4(self, init_image, next_image, action, p=3):
        ltn.diag(init_image, next_image, action)
        return Forall([init_image, next_image, action],
            #Sim(self.ltn_F_and_P.Dec(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(init_image)), self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(init_image)), self.ltn_F_and_P.Up(init_image)), next_image), p=p
            Sim(self.ltn_F_and_P.Dec(self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Front(init_image), action), self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Up(init_image), action), self.ltn_F_and_P.Up(init_image)), next_image), p=p 
        )

    def DecoderRuleA5(self, init_image, next_image, action, p=3):
        ltn.diag(init_image, next_image, action)
        return Forall([init_image, next_image, action],
            #Sim(self.ltn_F_and_P.Dec(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(init_image)), self.ltn_F_and_P.Right(init_image), self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(init_image))), next_image), p=p
            Sim(self.ltn_F_and_P.Dec(self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Front(init_image), action), self.ltn_F_and_P.Right(init_image), self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Right(init_image), action)), next_image), p=p 
        )
    
    def DecoderA0(self, init_image_a_0, action):
        return self.ltn_F_and_P.logic_models.dec(self.ltn_F_and_P.logic_models.front(init_image_a_0), self.ltn_F_and_P.logic_models.rot_change(self.ltn_F_and_P.logic_models.right(init_image_a_0), action), self.ltn_F_and_P.logic_models.front(init_image_a_0))
        #return self.ltn_F_and_P.logic_models.dec(self.ltn_F_and_P.logic_models.front(init_image_a_0), self.ltn_F_and_P.logic_models.rot_plus(self.ltn_F_and_P.logic_models.right(init_image_a_0)), self.ltn_F_and_P.logic_models.front(init_image_a_0))
    
    def DecoderA1(self, init_image_a_1, action):
        return self.ltn_F_and_P.logic_models.dec(self.ltn_F_and_P.logic_models.up(init_image_a_1), self.ltn_F_and_P.logic_models.rot_change(self.ltn_F_and_P.logic_models.right(init_image_a_1), action), self.ltn_F_and_P.logic_models.up(init_image_a_1))
        #return self.ltn_F_and_P.logic_models.dec(self.ltn_F_and_P.logic_models.up(init_image_a_1), self.ltn_F_and_P.logic_models.rot_minus(self.ltn_F_and_P.logic_models.right(init_image_a_1)), self.ltn_F_and_P.logic_models.up(init_image_a_1))
    
    def DecoderA2(self, init_image_a_2, action):
        return self.ltn_F_and_P.logic_models.dec(self.ltn_F_and_P.logic_models.right(init_image_a_2), self.ltn_F_and_P.logic_models.right(init_image_a_2), self.ltn_F_and_P.logic_models.rot_change(self.ltn_F_and_P.logic_models.up(init_image_a_2), action))
        #return self.ltn_F_and_P.logic_models.dec(self.ltn_F_and_P.logic_models.right(init_image_a_2), self.ltn_F_and_P.logic_models.right(init_image_a_2), self.ltn_F_and_P.logic_models.rot_plus(self.ltn_F_and_P.logic_models.up(init_image_a_2)))
    
    def DecoderA3(self, init_image_a_3, action):  
        return self.ltn_F_and_P.logic_models.dec(self.ltn_F_and_P.logic_models.front(init_image_a_3), self.ltn_F_and_P.logic_models.rot_change(self.ltn_F_and_P.logic_models.up(init_image_a_3), action), self.ltn_F_and_P.logic_models.front(init_image_a_3))
        #return self.ltn_F_and_P.logic_models.dec(self.ltn_F_and_P.logic_models.front(init_image_a_3), self.ltn_F_and_P.logic_models.rot_minus(self.ltn_F_and_P.logic_models.up(init_image_a_3)), self.ltn_F_and_P.logic_models.front(init_image_a_3))
    
    def DecoderA4(self, init_image_a_4, action):
        return self.ltn_F_and_P.logic_models.dec(self.ltn_F_and_P.logic_models.rot_change(self.ltn_F_and_P.logic_models.front(init_image_a_4), action), self.ltn_F_and_P.logic_models.rot_change(self.ltn_F_and_P.logic_models.up(init_image_a_4), action), self.ltn_F_and_P.logic_models.up(init_image_a_4))
        #return self.ltn_F_and_P.logic_models.dec(self.ltn_F_and_P.logic_models.rot_plus(self.ltn_F_and_P.logic_models.front(init_image_a_4)), self.ltn_F_and_P.logic_models.rot_plus(self.ltn_F_and_P.logic_models.up(init_image_a_4)), self.ltn_F_and_P.logic_models.up(init_image_a_4))
    
    def DecoderA5(self, init_image_a_5, action):
        return self.ltn_F_and_P.logic_models.dec(self.ltn_F_and_P.logic_models.rot_change(self.ltn_F_and_P.logic_models.front(init_image_a_5), action), self.ltn_F_and_P.logic_models.right(init_image_a_5), self.ltn_F_and_P.logic_models.rot_change(self.ltn_F_and_P.logic_models.right(init_image_a_5), action))
        #return self.ltn_F_and_P.logic_models.dec(self.ltn_F_and_P.logic_models.rot_minus(self.ltn_F_and_P.logic_models.front(init_image_a_5)), self.ltn_F_and_P.logic_models.right(init_image_a_5), self.ltn_F_and_P.logic_models.rot_minus(self.ltn_F_and_P.logic_models.right(init_image_a_5)))
    
    def get_reconstruction_based_on_actions(self, init_image, actions):
        init_image_a_0 = init_image[actions == 0]
        init_image_a_1 = init_image[actions == 1]       
        init_image_a_2 = init_image[actions == 2]
        init_image_a_3 = init_image[actions == 3]
        init_image_a_4 = init_image[actions == 4]
        init_image_a_5 = init_image[actions == 5]
        reconstructions = torch.zeros_like(init_image)
        if init_image_a_0.shape[0] != 0:
            reconstructions[actions == 0] = self.DecoderA0(init_image_a_0, actions[actions == 0])
        if init_image_a_1.shape[0] != 0:
            reconstructions[actions == 1] = self.DecoderA1(init_image_a_1, actions[actions == 1])
        if init_image_a_2.shape[0] != 0:
            reconstructions[actions == 2] = self.DecoderA2(init_image_a_2, actions[actions == 2])
        if init_image_a_3.shape[0] != 0:
            reconstructions[actions == 3] = self.DecoderA3(init_image_a_3, actions[actions == 3])
        if init_image_a_4.shape[0] != 0:
            reconstructions[actions == 4] = self.DecoderA4(init_image_a_4, actions[actions == 4])
        if init_image_a_5.shape[0] != 0:
            reconstructions[actions == 5] = self.DecoderA5(init_image_a_5, actions[actions == 5])
            
        return reconstructions

    

