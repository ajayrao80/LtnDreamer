import ltn
from ltn_qm.connectives_quantifiers import (Forall, Sim, And)

class EncoderRules:
    def __init__(self, ltn_F_and_P):
        # ltn functions and predicates
        self.ltn_F_and_P = ltn_F_and_P  

    def EncoderRuleA0(self, init_image, next_image, p=3):
        ltn.diag(init_image, next_image)
        return Forall([init_image, next_image],
            And(Sim(self.ltn_F_and_P.Front(init_image), self.ltn_F_and_P.Up(next_image)), Sim(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(init_image)), self.ltn_F_and_P.Right(next_image))) , p=p
        )

    def EncoderRuleA1(self, init_image, next_image, p=3):
        ltn.diag(init_image, next_image)
        return Forall([init_image, next_image],
            And(Sim(self.ltn_F_and_P.Up(init_image), self.ltn_F_and_P.Front(next_image)), Sim(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(init_image)), self.ltn_F_and_P.Right(next_image))) , p=p
        )

    def EncoderRuleA2(self, init_image, next_image, p=3):
        ltn.diag(init_image, next_image)
        return Forall([init_image, next_image],
            And(Sim(self.ltn_F_and_P.Right(init_image), self.ltn_F_and_P.Front(next_image)), Sim(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(init_image)), self.ltn_F_and_P.Up(next_image))) , p=p
        )

    def EncoderRuleA3(self, init_image, next_image, p=3):
        ltn.diag(init_image, next_image)
        return Forall([init_image, next_image],
            And(Sim(self.ltn_F_and_P.Front(init_image), self.ltn_F_and_P.Right(next_image)), Sim(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(init_image)), self.ltn_F_and_P.Up(next_image))) , p=p
        )

    def EncoderRuleA4(self, init_image, next_image, p=3):
        ltn.diag(init_image, next_image)
        return Forall([init_image, next_image],
            And(Sim(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(init_image)), self.ltn_F_and_P.Front(next_image)), Sim(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(init_image)), self.ltn_F_and_P.Right(next_image))) , p=p
        )

    def EncoderRuleA5(self, init_image, next_image, p=3):
        ltn.diag(init_image, next_image)
        return Forall([init_image, next_image],
            And(Sim(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(init_image)), self.ltn_F_and_P.Front(next_image)), Sim(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(init_image)), self.ltn_F_and_P.Right(next_image))) , p=p
        )


    

