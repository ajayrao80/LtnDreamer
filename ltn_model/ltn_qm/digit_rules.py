import ltn
from ltn_qm.connectives_quantifiers import (Forall, And, Implies, Not, Exists, Eq)

class DigitRules:
    def __init__(self, ltn_F_and_P):
        # ltn functions and predicates
        self.ltn_F_and_P = ltn_F_and_P  
    
    def get_digit_rules(self, init_image, next_image, digit_labels_f_init, digit_labels_r_init, digit_labels_u_init, digit_labels_f_next, digit_labels_r_next, digit_labels_u_next):
        #same_digit_rules = self.get_same_digit_rules(init_image_a_0, init_image_a_1, init_image_a_2, init_image_a_3, init_image_a_4, init_image_a_5, next_image_a_0, next_image_a_1, next_image_a_2, next_image_a_3, next_image_a_4, next_image_a_5)
        #different_digit_rules = self.get_different_digit_rules(init_image_a_0, init_image_a_1, init_image_a_2, init_image_a_3, init_image_a_4, init_image_a_5, next_image_a_0, next_image_a_1, next_image_a_2, next_image_a_3, next_image_a_4, next_image_a_5)
        #return same_digit_rules + different_digit_rules
        ltn.diag(init_image, digit_labels_f_init)
        digit_rules_f_init = Forall([init_image, digit_labels_f_init], Eq(self.ltn_F_and_P.DigitP(self.ltn_F_and_P.Front(init_image)), digit_labels_f_init))
        digit_rules_f_init_rp = Forall([init_image, digit_labels_f_init], Eq(self.ltn_F_and_P.DigitP(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(init_image))), digit_labels_f_init))
        digit_rules_f_init_rm = Forall([init_image, digit_labels_f_init], Eq(self.ltn_F_and_P.DigitP(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(init_image))), digit_labels_f_init))
        ltn.diag(init_image, digit_labels_r_init)
        digit_rules_r_init = Forall([init_image, digit_labels_r_init], Eq(self.ltn_F_and_P.DigitP(self.ltn_F_and_P.Right(init_image)), digit_labels_r_init))
        digit_rules_r_init_rp = Forall([init_image, digit_labels_r_init], Eq(self.ltn_F_and_P.DigitP(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(init_image))), digit_labels_r_init))
        digit_rules_r_init_rm = Forall([init_image, digit_labels_r_init], Eq(self.ltn_F_and_P.DigitP(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(init_image))), digit_labels_r_init))
        ltn.diag(init_image, digit_labels_u_init)
        digit_rules_u_init = Forall([init_image, digit_labels_u_init], Eq(self.ltn_F_and_P.DigitP(self.ltn_F_and_P.Up(init_image)), digit_labels_u_init))
        digit_rules_u_init_rp = Forall([init_image, digit_labels_u_init], Eq(self.ltn_F_and_P.DigitP(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(init_image))), digit_labels_u_init))
        digit_rules_u_init_rm = Forall([init_image, digit_labels_u_init], Eq(self.ltn_F_and_P.DigitP(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(init_image))), digit_labels_u_init))

        ltn.diag(next_image, digit_labels_f_next)
        digit_rules_f_next = Forall([next_image, digit_labels_f_next], Eq(self.ltn_F_and_P.DigitP(self.ltn_F_and_P.Front(next_image)), digit_labels_f_next))
        digit_rules_f_next_rp = Forall([next_image, digit_labels_f_next], Eq(self.ltn_F_and_P.DigitP(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(next_image))), digit_labels_f_next))
        digit_rules_f_next_rm = Forall([next_image, digit_labels_f_next], Eq(self.ltn_F_and_P.DigitP(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(next_image))), digit_labels_f_next))
        ltn.diag(next_image, digit_labels_r_next)
        digit_rules_r_next = Forall([next_image, digit_labels_r_next], Eq(self.ltn_F_and_P.DigitP(self.ltn_F_and_P.Right(next_image)), digit_labels_r_next))
        digit_rules_r_next_rp = Forall([next_image, digit_labels_r_next], Eq(self.ltn_F_and_P.DigitP(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(next_image))), digit_labels_r_next))
        digit_rules_r_next_rm = Forall([next_image, digit_labels_r_next], Eq(self.ltn_F_and_P.DigitP(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(next_image))), digit_labels_r_next))
        ltn.diag(next_image, digit_labels_u_next)
        digit_rules_u_next = Forall([next_image, digit_labels_u_next], Eq(self.ltn_F_and_P.DigitP(self.ltn_F_and_P.Up(next_image)), digit_labels_u_next))
        digit_rules_u_next_rp = Forall([next_image, digit_labels_u_next], Eq(self.ltn_F_and_P.DigitP(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(next_image))), digit_labels_u_next))
        digit_rules_u_next_rm = Forall([next_image, digit_labels_u_next], Eq(self.ltn_F_and_P.DigitP(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(next_image))), digit_labels_u_next))

        return And(And(And(And(And(And(And(digit_rules_f_init, digit_rules_r_init), digit_rules_u_init), digit_rules_f_next), digit_rules_r_next), digit_rules_u_next),
                   And(And(And(And(And(digit_rules_f_init_rp, digit_rules_r_init_rp), digit_rules_u_init_rp), digit_rules_f_init_rm), digit_rules_r_init_rm), digit_rules_u_init_rm)),
                   And(And(And(And(And(digit_rules_f_next_rp, digit_rules_r_next_rp), digit_rules_u_next_rp), digit_rules_f_next_rm), digit_rules_r_next_rm), digit_rules_u_next_rm))
    
    def get_same_digit_rules(self, init_image_a_0, init_image_a_1, init_image_a_2, init_image_a_3, init_image_a_4, init_image_a_5, next_image_a_0, next_image_a_1, next_image_a_2, next_image_a_3, next_image_a_4, next_image_a_5):
        rules = []
        for P in self.ltn_F_and_P.DigitP:
            rule_1 = self.SameDigitRuleA0(init_image_a_0, next_image_a_0, P) if init_image_a_0 is not None else None
            rule_2 = self.SameDigitRuleA1(init_image_a_1, next_image_a_1, P) if init_image_a_1 is not None else None
            rule_3 = self.SameDigitRuleA2(init_image_a_2, next_image_a_2, P) if init_image_a_2 is not None else None
            rule_4 = self.SameDigitRuleA3(init_image_a_3, next_image_a_3, P) if init_image_a_3 is not None else None
            rule_5 = self.SameDigitRuleA4(init_image_a_4, next_image_a_4, P) if init_image_a_4 is not None else None
            rule_6 = self.SameDigitRuleA5(init_image_a_5, next_image_a_5, P) if init_image_a_5 is not None else None
            rules.append(rule_1)
            rules.append(rule_2)
            rules.append(rule_3)
            rules.append(rule_4)
            rules.append(rule_5)
            rules.append(rule_6)
        
        rules = [rule for rule in rules if rule is not None]
        return rules
        
    def get_different_digit_rules(self, init_image_a_0, init_image_a_1, init_image_a_2, init_image_a_3, init_image_a_4, init_image_a_5, next_image_a_0, next_image_a_1, next_image_a_2, next_image_a_3, next_image_a_4, next_image_a_5):
        rules = []
        for P in self.ltn_F_and_P.DigitP:
            rule_1 = self.DifferentRuleA0(init_image_a_0, next_image_a_0, P) if init_image_a_0 is not None else None
            rule_2 = self.DifferentRuleA1(init_image_a_1, next_image_a_1, P) if init_image_a_1 is not None else None
            rule_3 = self.DifferentRuleA2(init_image_a_2, next_image_a_2, P) if init_image_a_2 is not None else None
            rule_4 = self.DifferentRuleA3(init_image_a_3, next_image_a_3, P) if init_image_a_3 is not None else None
            rule_5 = self.DifferentRuleA4(init_image_a_4, next_image_a_4, P) if init_image_a_4 is not None else None
            rule_6 = self.DifferentRuleA5(init_image_a_5, next_image_a_5, P) if init_image_a_5 is not None else None
            rules.append(rule_1)
            rules.append(rule_2)
            rules.append(rule_3)
            rules.append(rule_4)
            rules.append(rule_5)
            rules.append(rule_6)
        
        rules = [rule for rule in rules if rule is not None]
        return rules
    
    def general_digit_constraints(self, init_image, next_image):
        constraints = []
        for P in self.ltn_F_and_P.DigitP:
            constraint_1 = Exists([init_image], P(self.ltn_F_and_P.Front(init_image)))
            constraint_2 = Exists([init_image], P(self.ltn_F_and_P.Right(init_image)))
            constraint_3 = Exists([init_image], P(self.ltn_F_and_P.Up(init_image)))
            constraint_4 = Exists([next_image], P(self.ltn_F_and_P.Front(next_image)))
            constraint_5 = Exists([next_image], P(self.ltn_F_and_P.Right(next_image)))
            constraint_6 = Exists([next_image], P(self.ltn_F_and_P.Up(next_image)))
            constraints.append(constraint_1)
            constraints.append(constraint_2)
            constraints.append(constraint_3)
            constraints.append(constraint_4)
            constraints.append(constraint_5)
            constraints.append(constraint_6)
        
        rules = [rule for rule in constraints if rule is not None]
        return rules

    # TODO: Write it in a compact manner
    def SameDigitRuleA0(self, image_1, image_2, P, p=3):
        ltn.diag(image_1, image_2)
        rule_f_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), P(self.ltn_F_and_P.Up(image_2))), p=p)
        rule_f_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_2)))), p=p)
        rule_f_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_2)))), p=p)

        rule_r_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_1))), P(self.ltn_F_and_P.Right(image_2))), p=p)
        rule_r_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_1))), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_r_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_1))), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_2)))), p=p)

        rule_f_rot_1 = Forall([image_1], Implies(P(self.ltn_F_and_P.Front(image_1)), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_1)))), p=p)
        rule_f_rot_2 = Forall([image_1], Implies(P(self.ltn_F_and_P.Front(image_1)), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_1)))), p=p)
        rule_r_rot_1 = Forall([image_1], Implies(P(self.ltn_F_and_P.Right(image_1)), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_1)))), p=p)
        rule_r_rot_2 = Forall([image_1], Implies(P(self.ltn_F_and_P.Right(image_1)), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_1)))), p=p)

        rule_f = And(And(rule_f_1, rule_f_2), rule_f_3)
        rule_r = And(And(rule_r_1, rule_r_2), rule_r_3)
        rule_f_rot = And(rule_f_rot_1, rule_f_rot_2)
        rule_r_rot = And(rule_r_rot_1, rule_r_rot_2)

        return And(And(And(rule_f, rule_r), rule_f_rot), rule_r_rot)

    def SameDigitRuleA1(self, image_1, image_2, P, p=3):
        ltn.diag(image_1, image_2)
        rule_u_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), P(self.ltn_F_and_P.Front(image_2))), p=p)
        rule_u_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_2)))), p=p)
        rule_u_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_2)))), p=p)

        rule_r_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_1))), P(self.ltn_F_and_P.Right(image_2))), p=p)
        rule_r_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_1))), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_r_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_1))), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_2)))), p=p)

        rule_u_rot_1 = Forall([image_1], Implies(P(self.ltn_F_and_P.Up(image_1)), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_1)))), p=p)
        rule_u_rot_2 = Forall([image_1], Implies(P(self.ltn_F_and_P.Up(image_1)), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_1)))), p=p)
        rule_r_rot_1 = Forall([image_1], Implies(P(self.ltn_F_and_P.Right(image_1)), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_1)))), p=p)
        rule_r_rot_2 = Forall([image_1], Implies(P(self.ltn_F_and_P.Right(image_1)), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_1)))), p=p)

        rule_u = And(And(rule_u_1, rule_u_2), rule_u_3)
        rule_r = And(And(rule_r_1, rule_r_2), rule_r_3)
        rule_u_rot = And(rule_u_rot_1, rule_u_rot_2)
        rule_r_rot = And(rule_r_rot_1, rule_r_rot_2)
        return And(And(And(rule_u, rule_r), rule_u_rot), rule_r_rot)

    def SameDigitRuleA2(self, image_1, image_2, P, p=3):
        ltn.diag(image_1, image_2)
        rule_r_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), P(self.ltn_F_and_P.Front(image_2))), p=p)
        rule_r_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_2)))), p=p)
        rule_r_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_2)))), p=p)

        rule_u_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_1))), P(self.ltn_F_and_P.Up(image_2))), p=p)
        rule_u_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_1))), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_2)))), p=p)
        rule_u_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_1))), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_2)))), p=p)

        rule_u_rot_1 = Forall([image_1], Implies(P(self.ltn_F_and_P.Up(image_1)), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_1)))), p=p)
        rule_u_rot_2 = Forall([image_1], Implies(P(self.ltn_F_and_P.Up(image_1)), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_1)))), p=p)
        rule_r_rot_1 = Forall([image_1], Implies(P(self.ltn_F_and_P.Right(image_1)), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_1)))), p=p)
        rule_r_rot_2 = Forall([image_1], Implies(P(self.ltn_F_and_P.Right(image_1)), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_1)))), p=p)

        rule_u = And(And(rule_u_1, rule_u_2), rule_u_3)
        rule_r = And(And(rule_r_1, rule_r_2), rule_r_3)
        rule_u_rot = And(rule_u_rot_1, rule_u_rot_2)
        rule_r_rot = And(rule_r_rot_1, rule_r_rot_2)
        return And(And(And(rule_u, rule_r), rule_u_rot), rule_r_rot)

    def SameDigitRuleA3(self, image_1, image_2, P, p=3):
        ltn.diag(image_1, image_2)
        rule_f_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), P(self.ltn_F_and_P.Right(image_2))), p=p)
        rule_f_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_f_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_2)))), p=p)

        rule_u_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_1))), P(self.ltn_F_and_P.Up(image_2))), p=p)
        rule_u_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_1))), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_2)))), p=p)
        rule_u_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_1))), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_2)))), p=p)

        rule_u_rot_1 = Forall([image_1], Implies(P(self.ltn_F_and_P.Up(image_1)), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_1)))), p=p)
        rule_u_rot_2 = Forall([image_1], Implies(P(self.ltn_F_and_P.Up(image_1)), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_1)))), p=p)
        rule_r_rot_1 = Forall([image_1], Implies(P(self.ltn_F_and_P.Right(image_1)), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_1)))), p=p)
        rule_r_rot_2 = Forall([image_1], Implies(P(self.ltn_F_and_P.Right(image_1)), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_1)))), p=p)

        rule_r = And(And(rule_f_1, rule_f_2), rule_f_3)
        rule_u = And(And(rule_u_1, rule_u_2), rule_u_3)
        rule_u_rot = And(rule_u_rot_1, rule_u_rot_2)
        rule_r_rot = And(rule_r_rot_1, rule_r_rot_2)
        return And(And(And(rule_u, rule_r), rule_u_rot), rule_r_rot)

    def SameDigitRuleA4(self, image_1, image_2, P, p=3):
        ltn.diag(image_1, image_2)
        rule_f_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_1))), P(self.ltn_F_and_P.Front(image_2))), p=p)
        rule_f_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_1))), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_2)))), p=p)
        rule_f_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_1))), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_2)))), p=p)

        rule_u_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_1))), P(self.ltn_F_and_P.Right(image_2))), p=p)
        rule_u_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_1))), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_u_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_1))), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_2)))), p=p)

        rule_u_rot_1 = Forall([image_1], Implies(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_1))), P(self.ltn_F_and_P.Up(image_1))), p=p)
        rule_u_rot_2 = Forall([image_1], Implies(P(self.ltn_F_and_P.Up(image_1)), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_1)))), p=p)
        rule_f_rot_1 = Forall([image_1], Implies(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_1))), P(self.ltn_F_and_P.Front(image_1))), p=p)
        rule_f_rot_2 = Forall([image_1], Implies(P(self.ltn_F_and_P.Front(image_1)), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_1)))), p=p)

        rule_f = And(And(rule_f_1, rule_f_2), rule_f_3)
        rule_u = And(And(rule_u_1, rule_u_2), rule_u_3)
        rule_u_rot = And(rule_u_rot_1, rule_u_rot_2)
        rule_f_rot = And(rule_f_rot_1, rule_f_rot_2)
        return And(And(And(rule_u, rule_f), rule_u_rot), rule_f_rot)

    def SameDigitRuleA5(self, image_1, image_2, P, p=3):
        ltn.diag(image_1, image_2)
        rule_f_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_1))), P(self.ltn_F_and_P.Front(image_2))), p=p)
        rule_f_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_1))), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_2)))), p=p)
        rule_f_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_1))), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_2)))), p=p)

        rule_r_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_1))), P(self.ltn_F_and_P.Up(image_2))), p=p)
        rule_r_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_1))), P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_2)))), p=p)
        rule_r_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_1))), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_2)))), p=p)

        rule_r_rot_1 = Forall([image_1], Implies(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_1))), P(self.ltn_F_and_P.Right(image_1))), p=p)
        rule_r_rot_2 = Forall([image_1], Implies(P(self.ltn_F_and_P.Right(image_1)), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_1)))), p=p)
        rule_f_rot_1 = Forall([image_1], Implies(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_1))), P(self.ltn_F_and_P.Front(image_1))), p=p)
        rule_f_rot_2 = Forall([image_1], Implies(P(self.ltn_F_and_P.Front(image_1)), P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_1)))), p=p)

        rule_f = And(And(rule_f_1, rule_f_2), rule_f_3)
        rule_r = And(And(rule_r_1, rule_r_2), rule_r_3)
        rule_r_rot = And(rule_r_rot_1, rule_r_rot_2)
        rule_f_rot = And(rule_f_rot_1, rule_f_rot_2)
        return And(And(And(rule_r, rule_f), rule_r_rot), rule_f_rot)
    
    def DifferentRuleA0(self, image_1, image_2, P, p=2):
        ltn.diag(image_1, image_2)
        rule_f_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.Front(image_2)))), p=p)
        rule_f_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_f_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_f_4 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_f_5 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_f_6 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_2))))), p=p)

        rule_r_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.Front(image_2)))), p=p)
        rule_r_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_r_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_r_4 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.Up(image_2)))), p=p)
        rule_r_5 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_2))))), p=p)
        rule_r_6 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_2))))), p=p)

        rule_u_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.Front(image_2)))), p=p)
        rule_u_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_u_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_2))))), p=p)

        rule_u_4 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_u_5 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_u_6 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_2))))), p=p)

        rule_u_7 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.Up(image_2)))), p=p)
        rule_u_8 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_2))))), p=p)
        rule_u_9 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_2))))), p=p)

        F = And(And(And(And(And(rule_f_1,rule_f_2), rule_f_3), rule_f_4), rule_f_5), rule_f_6)
        R = And(And(And(And(And(rule_r_1, rule_r_2), rule_r_3), rule_r_4), rule_r_5), rule_r_6)
        U = And(And(And(And(And(And(And(And(rule_u_1, rule_u_2), rule_u_3), rule_u_4), rule_u_5),rule_u_6), rule_u_7), rule_u_8), rule_u_9)
        return And(And(F, R), U)

    def DifferentRuleA1(self, image_1, image_2, P, p=2):
        ltn.diag(image_1, image_2)
        rule_f_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.Front(image_2)))), p=p)
        rule_f_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_f_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_f_4 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_f_5 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_f_6 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_f_7 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.Up(image_2)))), p=p)
        rule_f_8 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_2))))), p=p)
        rule_f_9 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_2))))), p=p)

        rule_r_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.Front(image_2)))), p=p)
        rule_r_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_r_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_r_4 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.Up(image_2)))), p=p)
        rule_r_5 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_2))))), p=p)
        rule_r_6 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_2))))), p=p)

        rule_u_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_u_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_u_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_u_4 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.Up(image_2)))), p=p)
        rule_u_5 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_2))))), p=p)
        rule_u_6 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_2))))), p=p)

        F = And(And(And(And(And(And(And(And(rule_f_1,rule_f_2), rule_f_3), rule_f_4), rule_f_5), rule_f_6), rule_f_7), rule_f_8), rule_f_9)
        R = And(And(And(And(And(rule_r_1, rule_r_2), rule_r_3), rule_r_4), rule_r_5), rule_r_6)
        U = And(And(And(And(And(rule_u_1, rule_u_2), rule_u_3), rule_u_4), rule_u_5), rule_u_6)
        return And(And(F, R), U)
    
    def DifferentRuleA2(self, image_1, image_2, P, p=2):
        ltn.diag(image_1, image_2)
        rule_f_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.Front(image_2)))), p=p)
        rule_f_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_f_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_f_4 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_f_5 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_f_6 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_f_7 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.Up(image_2)))), p=p)
        rule_f_8 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_2))))), p=p)
        rule_f_9 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_2))))), p=p)

        rule_r_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_r_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_r_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_r_4 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.Up(image_2)))), p=p)
        rule_r_5 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_2))))), p=p)
        rule_r_6 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_2))))), p=p)

        rule_u_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.Front(image_2)))), p=p)
        rule_u_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_u_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_u_4 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_u_5 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_u_6 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_2))))), p=p)

        F = And(And(And(And(And(And(And(And(rule_f_1,rule_f_2), rule_f_3), rule_f_4), rule_f_5), rule_f_6), rule_f_7), rule_f_8), rule_f_9)
        R = And(And(And(And(And(rule_r_1, rule_r_2), rule_r_3), rule_r_4), rule_r_5), rule_r_6)
        U = And(And(And(And(And(rule_u_1, rule_u_2), rule_u_3), rule_u_4), rule_u_5),rule_u_6)

        return And(And(F, R), U)

    def DifferentRuleA3(self, image_1, image_2, P, p=2):
        ltn.diag(image_1, image_2)
        rule_f_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.Front(image_2)))), p=p)
        rule_f_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_f_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_f_4 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.Up(image_2)))), p=p)
        rule_f_5 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_2))))), p=p)
        rule_f_6 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_2))))), p=p)

        rule_r_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.Front(image_2)))), p=p)
        rule_r_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_r_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_r_4 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_r_5 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_r_6 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_r_7 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.Up(image_2)))), p=p)
        rule_r_8 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_2))))), p=p)
        rule_r_9 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_2))))), p=p)

        rule_u_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.Front(image_2)))), p=p)
        rule_u_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_u_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_u_4 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_u_5 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_u_6 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_2))))), p=p)

        F = And(And(And(And(And(rule_f_1,rule_f_2), rule_f_3), rule_f_4), rule_f_5), rule_f_6)
        R = And(And(And(And(And(And(And(And(rule_r_1, rule_r_2), rule_r_3), rule_r_4), rule_r_5), rule_r_6), rule_r_7), rule_r_8), rule_r_9)
        U = And(And(And(And(And(rule_u_1, rule_u_2), rule_u_3), rule_u_4), rule_u_5),rule_u_6)

        return And(And(F, R), U)

    def DifferentRuleA4(self, image_1, image_2, P, p=2):
        ltn.diag(image_1, image_2)
        rule_f_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_f_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_f_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_f_4 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.Up(image_2)))), p=p)
        rule_f_5 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_2))))), p=p)
        rule_f_6 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_2))))), p=p)

        rule_r_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.Front(image_2)))), p=p)
        rule_r_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_r_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_r_4 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_r_5 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_r_6 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_r_7 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.Up(image_2)))), p=p)
        rule_r_8 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_2))))), p=p)
        rule_r_9 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_2))))), p=p)

        rule_u_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.Front(image_2)))), p=p)
        rule_u_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_u_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_u_4 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.Up(image_2)))), p=p)
        rule_u_5 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_2))))), p=p)
        rule_u_6 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_2))))), p=p)

        F = And(And(And(And(And(rule_f_1,rule_f_2), rule_f_3), rule_f_4), rule_f_5), rule_f_6)
        R = And(And(And(And(And(And(And(And(rule_r_1, rule_r_2), rule_r_3), rule_r_4), rule_r_5), rule_r_6), rule_r_7), rule_r_8), rule_r_9)
        U = And(And(And(And(And(rule_u_1, rule_u_2), rule_u_3), rule_u_4), rule_u_5),rule_u_6)
    
        return And(And(F, R), U)

    def DifferentRuleA5(self, image_1, image_2, P, p=2):
        ltn.diag(image_1, image_2)
        rule_f_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_f_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_f_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_f_4 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.Up(image_2)))), p=p)
        rule_f_5 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_2))))), p=p)
        rule_f_6 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Front(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_2))))), p=p)

        rule_r_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_r_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_r_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_r_4 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_r_5 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_r_6 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Right(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_2))))), p=p)

        rule_u_1 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.Front(image_2)))), p=p)
        rule_u_2 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_u_3 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Front(image_2))))), p=p)
        rule_u_4 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.Right(image_2)))), p=p)
        rule_u_5 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_u_6 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Right(image_2))))), p=p)
        rule_u_7 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.Up(image_2)))), p=p)
        rule_u_8 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotPlus(self.ltn_F_and_P.Up(image_2))))), p=p)
        rule_u_9 = Forall([image_1, image_2], Implies(P(self.ltn_F_and_P.Up(image_1)), Not(P(self.ltn_F_and_P.RotMinus(self.ltn_F_and_P.Up(image_2))))), p=p)

        F = And(And(And(And(And(rule_f_1,rule_f_2), rule_f_3), rule_f_4), rule_f_5), rule_f_6)
        R = And(And(And(And(And(rule_r_1, rule_r_2), rule_r_3), rule_r_4), rule_r_5), rule_r_6)
        U = And(And(And(And(And(And(And(And(rule_u_1, rule_u_2), rule_u_3), rule_u_4), rule_u_5),rule_u_6), rule_u_7), rule_u_8), rule_u_9)
        
        return And(And(F, R), U)
