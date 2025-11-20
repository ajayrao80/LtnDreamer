import ltn
from ltn_model.ltn_qm.connectives_quantifiers import (Forall, Sim)

class DecoderRules:
    def __init__(self, ltn_F_and_P):
        # ltn functions and predicates
        self.ltn_F_and_P = ltn_F_and_P  

    def DecoderRuleA0(self, init_image, next_image, p=3):
        ltn.diag(init_image, next_image)
        return Forall([init_image, next_image],
            Sim(self.ltn_F_and_P.Dec(self.ltn_F_and_P.Front(init_image), self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(init_image)), self.ltn_F_and_P.Front(init_image)), next_image), p=p 
        )

    def DecoderRuleA1(self,init_image, next_image, p=3):
        ltn.diag(init_image, next_image)
        return Forall([init_image, next_image],
            Sim(self.ltn_F_and_P.Dec(self.ltn_F_and_P.Up(init_image), self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(init_image)), self.ltn_F_and_P.Up(init_image)), next_image), p=p 
        )

    def DecoderRuleA2(self, init_image, next_image, p=3):
        ltn.diag(init_image, next_image)
        return Forall([init_image, next_image],
            Sim(self.ltn_F_and_P.Dec(self.ltn_F_and_P.Right(init_image), self.ltn_F_and_P.Right(init_image), self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(init_image))), next_image), p=p 
        )

    def DecoderRuleA3(self, init_image, next_image, p=3):
        ltn.diag(init_image, next_image)
        return Forall([init_image, next_image],
            Sim(self.ltn_F_and_P.Dec(self.ltn_F_and_P.Front(init_image), self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(init_image)), self.ltn_F_and_P.Front(init_image)), next_image), p=p 
        )

    def DecoderRuleA4(self, init_image, next_image, p=3):
        ltn.diag(init_image, next_image)
        return Forall([init_image, next_image],
            Sim(self.ltn_F_and_P.Dec(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(init_image)), self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(init_image)), self.ltn_F_and_P.Up(init_image)), next_image), p=p 
        )

    def DecoderRuleA5(self, init_image, next_image, p=3):
        ltn.diag(init_image, next_image)
        return Forall([init_image, next_image],
            Sim(self.ltn_F_and_P.Dec(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(init_image)), self.ltn_F_and_P.Right(init_image), self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(init_image))), next_image), p=p 
        )
    

