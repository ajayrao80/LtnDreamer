import ltn
from ltn_model.ltn_qm.connectives_quantifiers import (Forall, Eq, And)

class EncoderRules:
    def __init__(self, ltn_F_and_P):
        # ltn functions and predicates
        self.ltn_F_and_P = ltn_F_and_P  

    def EncoderRuleA0(self, init_image, next_image, action, p=5):
        ltn.diag(init_image, next_image, action)
        return Forall([init_image, next_image, action],
            #And(Sim(self.ltn_F_and_P.Front(init_image), self.ltn_F_and_P.Up(next_image)), Sim(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(init_image)), self.ltn_F_and_P.Right(next_image))) , p=p
            And(Eq(self.ltn_F_and_P.Front(init_image), self.ltn_F_and_P.Up(next_image)), Eq(self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Right(init_image), action), self.ltn_F_and_P.Right(next_image))), p=p
        )

    def EncoderRuleA1(self, init_image, next_image, action, p=5):
        ltn.diag(init_image, next_image, action)
        return Forall([init_image, next_image, action],
            #And(Sim(self.ltn_F_and_P.Up(init_image), self.ltn_F_and_P.Front(next_image)), Sim(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(init_image)), self.ltn_F_and_P.Right(next_image))) , p=p
            And(Eq(self.ltn_F_and_P.Up(init_image), self.ltn_F_and_P.Front(next_image)), Eq(self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Right(init_image), action), self.ltn_F_and_P.Right(next_image))), p=p 
        )

    def EncoderRuleA2(self, init_image, next_image, action, p=5):
        ltn.diag(init_image, next_image, action)
        return Forall([init_image, next_image, action],
            #And(Sim(self.ltn_F_and_P.Right(init_image), self.ltn_F_and_P.Front(next_image)), Sim(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(init_image)), self.ltn_F_and_P.Up(next_image))) , p=p
            And(Eq(self.ltn_F_and_P.Right(init_image), self.ltn_F_and_P.Front(next_image)), Eq(self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Up(init_image), action), self.ltn_F_and_P.Up(next_image))), p=p
        )

    def EncoderRuleA3(self, init_image, next_image, action, p=5):
        ltn.diag(init_image, next_image, action)
        return Forall([init_image, next_image, action],
            #And(Sim(self.ltn_F_and_P.Front(init_image), self.ltn_F_and_P.Right(next_image)), Sim(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(init_image)), self.ltn_F_and_P.Up(next_image))) , p=p
            And(Eq(self.ltn_F_and_P.Front(init_image), self.ltn_F_and_P.Right(next_image)), Eq(self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Up(init_image), action), self.ltn_F_and_P.Up(next_image))), p=p
        )

    def EncoderRuleA4(self, init_image, next_image, action, p=5):
        ltn.diag(init_image, next_image, action)
        return Forall([init_image, next_image, action],
            #And(Sim(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(init_image)), self.ltn_F_and_P.Front(next_image)), Sim(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(init_image)), self.ltn_F_and_P.Right(next_image))) , p=p
            And(Eq(self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Front(init_image), action), self.ltn_F_and_P.Front(next_image)), Eq(self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Up(init_image), action), self.ltn_F_and_P.Right(next_image))), p=p
        )

    def EncoderRuleA5(self, init_image, next_image, action, p=5):
        ltn.diag(init_image, next_image, action)
        return Forall([init_image, next_image, action],
            #And(Sim(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(init_image)), self.ltn_F_and_P.Front(next_image)), Sim(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(init_image)), self.ltn_F_and_P.Right(next_image))) , p=p
            And(Eq(self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Front(init_image), action), self.ltn_F_and_P.Front(next_image)), Eq(self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Up(init_image), action), self.ltn_F_and_P.Right(next_image))), p=p
        )
    
    def EncoderEqRuleA0(self, init_image, r, u, action, p=5):
        ltn.diag(init_image, r, u, action)
        return Forall([init_image, r, u, action],
            And(Eq(self.ltn_F_and_P.Front(init_image), u), Eq(self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Right(init_image), action), r)), p=p
        )
    
    def EncoderEqRuleA1(self, init_image, f, r, action, p=5):
        ltn.diag(init_image, f, r, action)
        return Forall([init_image, f, r, action],
            And(Eq(self.ltn_F_and_P.Up(init_image), f), Eq(self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Right(init_image), action), r)), p=p 
        )
    
    def EncoderEqRuleA2(self, init_image, f, u, action, p=5):
        ltn.diag(init_image, f, u, action)
        return Forall([init_image, f, u, action],
            And(Eq(self.ltn_F_and_P.Right(init_image), f), Eq(self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Up(init_image), action), u)), p=p
        )
    
    def EncoderEqRuleA3(self, init_image, r, u, action, p=5):
        ltn.diag(init_image, r, u, action)
        return Forall([init_image, r, u, action],
            And(Eq(self.ltn_F_and_P.Front(init_image), r), Eq(self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Up(init_image), action), u)), p=p
        )
    
    def EncoderEqRuleA4(self, init_image, f, r, action, p=5):
        ltn.diag(init_image, f, r, action)
        return Forall([init_image, f, r, action],
            And(Eq(self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Front(init_image), action), f), Eq(self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Up(init_image), action), r)), p=p
        )
    
    def EncoderEqRuleA5(self, init_image, f, r, action, p=5):
        ltn.diag(init_image, f, r, action)
        return Forall([init_image, f, r, action],
            And(Eq(self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Front(init_image), action), f), Eq(self.ltn_F_and_P.RotChange(self.ltn_F_and_P.Up(init_image), action), r)), p=p
        )
    




    

 