from ltn_model.modules.model import LTNModel
from ltn_model.utils.utils import cosine_similarity
import ltn
from ltn_model.ltn_qm.connectives_quantifiers import (And, Or, Not, Forall, Exists, Implies, SatAgg, Sim)
from ltn_model.ltn_qm.decoder_rules import DecoderRules
from ltn_model.ltn_qm.encoder_rules import EncoderRules
from ltn_model.ltn_qm.digit_rules import DigitRules
import torch

class LTNRules:
    def __init__(self, ltn_model=None):
        self.logic_models = LTNModel() if ltn_model is None else ltn_model
        # Functions and predicates ------------------------------------------
        self.Front = ltn.Function(model=self.logic_models.front)
        self.Right = ltn.Function(model=self.logic_models.right)
        self.Up = ltn.Function(model=self.logic_models.up)
        self.Dec = ltn.Function(model=self.logic_models.dec)
        self.RotChange = ltn.Function(model=self.logic_models.rot_change)
        #self.RotPlus = ltn.Function(model=self.logic_models.rot_plus)
        #self.RotMinus = ltn.Function(model=self.logic_models.rot_minus)
        #self.DigitP = ltn.Predicate(model=self.logic_models.digits) #[ltn.Predicate(model=self.logic_models.digits[i]) for i in range(len(self.logic_models.digits))]

        self.decoder_constraints = DecoderRules(self)
        self.encoder_constraints = EncoderRules(self)
        #self.digit_constraints = DigitRules(self)
        #self.num_digit_classes = 10
    
    def compute_sat(self, init_image, actions_, next_image, digits_labels_init=None, digits_labels_next=None):
        # cube 1 -----------------------------------------------------------
        init_image_a_0 = ltn.Variable("init_image_a_0", init_image[actions_.squeeze(1) == 0]) if init_image[actions_.squeeze(1) == 0].shape[0] != 0 else None
        init_image_a_1 = ltn.Variable("init_image_a_1", init_image[actions_.squeeze(1) == 1]) if init_image[actions_.squeeze(1) == 1].shape[0] != 0 else None
        init_image_a_2 = ltn.Variable("init_image_a_2", init_image[actions_.squeeze(1) == 2]) if init_image[actions_.squeeze(1) == 2].shape[0] != 0 else None
        init_image_a_3 = ltn.Variable("init_image_a_3", init_image[actions_.squeeze(1) == 3]) if init_image[actions_.squeeze(1) == 3].shape[0] != 0 else None
        init_image_a_4 = ltn.Variable("init_image_a_4", init_image[actions_.squeeze(1) == 4]) if init_image[actions_.squeeze(1) == 4].shape[0] != 0 else None
        init_image_a_5 = ltn.Variable("init_image_a_5", init_image[actions_.squeeze(1) == 5]) if init_image[actions_.squeeze(1) == 5].shape[0] != 0 else None

        # cube 2 -----------------------------------------------------------
        next_image_a_0 = ltn.Variable("next_image_a_0", next_image[actions_.squeeze(1) == 0]) if next_image[actions_.squeeze(1) == 0].shape[0] != 0 else None
        next_image_a_1 = ltn.Variable("next_image_a_1", next_image[actions_.squeeze(1) == 1]) if next_image[actions_.squeeze(1) == 1].shape[0] != 0 else None
        next_image_a_2 = ltn.Variable("next_image_a_2", next_image[actions_.squeeze(1) == 2]) if next_image[actions_.squeeze(1) == 2].shape[0] != 0 else None
        next_image_a_3 = ltn.Variable("next_image_a_3", next_image[actions_.squeeze(1) == 3]) if next_image[actions_.squeeze(1) == 3].shape[0] != 0 else None
        next_image_a_4 = ltn.Variable("next_image_a_4", next_image[actions_.squeeze(1) == 4]) if next_image[actions_.squeeze(1) == 4].shape[0] != 0 else None
        next_image_a_5 = ltn.Variable("next_image_a_5", next_image[actions_.squeeze(1) == 5]) if next_image[actions_.squeeze(1) == 5].shape[0] != 0 else None

        actions_a_0 = ltn.Variable("actions_a_0", actions_[actions_.squeeze(1) == 0]) if actions_[actions_.squeeze(1) == 0].shape[0] != 0 else None
        actions_a_1 = ltn.Variable("actions_a_1", actions_[actions_.squeeze(1) == 1]) if actions_[actions_.squeeze(1) == 1].shape[0] != 0 else None
        actions_a_2 = ltn.Variable("actions_a_2", actions_[actions_.squeeze(1) == 2]) if actions_[actions_.squeeze(1) == 2].shape[0] != 0 else None
        actions_a_3 = ltn.Variable("actions_a_3", actions_[actions_.squeeze(1) == 3]) if actions_[actions_.squeeze(1) == 3].shape[0] != 0 else None
        actions_a_4 = ltn.Variable("actions_a_4", actions_[actions_.squeeze(1) == 4]) if actions_[actions_.squeeze(1) == 4].shape[0] != 0 else None
        actions_a_5 = ltn.Variable("actions_a_5", actions_[actions_.squeeze(1) == 5]) if actions_[actions_.squeeze(1) == 5].shape[0] != 0 else None
        reconstruction_axioms_based_on_actions_1 = self.get_encoder_rules(init_image_a_0, init_image_a_1, init_image_a_2, init_image_a_3, init_image_a_4, init_image_a_5, next_image_a_0, next_image_a_1, next_image_a_2, next_image_a_3, next_image_a_4, next_image_a_5, actions_a_0, actions_a_1, actions_a_2, actions_a_3, actions_a_4, actions_a_5)
        reconstruction_axioms_based_on_actions_2 = self.get_decoder_rules(init_image_a_0, init_image_a_1, init_image_a_2, init_image_a_3, init_image_a_4, init_image_a_5, next_image_a_0, next_image_a_1, next_image_a_2, next_image_a_3, next_image_a_4, next_image_a_5, actions_a_0, actions_a_1, actions_a_2, actions_a_3, actions_a_4, actions_a_5)

        if digits_labels_init is not None:
            init_image_ = ltn.Variable("init_image_", init_image)
            next_image_ = ltn.Variable("next_image_", next_image)
            digit_labels_f_init = ltn.Variable("digit_labels_f_init", torch.nn.functional.one_hot(digits_labels_init[:, 0].long(), self.num_digit_classes))
            digit_labels_r_init = ltn.Variable("digit_labels_r_init", torch.nn.functional.one_hot(digits_labels_init[:, 2].long(), self.num_digit_classes)) 
            digit_labels_u_init = ltn.Variable("digit_labels_u_init", torch.nn.functional.one_hot(digits_labels_init[:, 1].long(), self.num_digit_classes)) 
            digit_labels_f_next = ltn.Variable("digit_labels_f_next", torch.nn.functional.one_hot(digits_labels_next[:, 0].long(), self.num_digit_classes))
            digit_labels_r_next = ltn.Variable("digit_labels_r_next", torch.nn.functional.one_hot(digits_labels_next[:, 2].long(), self.num_digit_classes))
            digit_labels_u_next = ltn.Variable("digit_labels_u_next", torch.nn.functional.one_hot(digits_labels_next[:, 1].long(), self.num_digit_classes))

            digit_classification_rules = self.get_digit_rules(init_image_, next_image_, digit_labels_f_init, digit_labels_r_init, digit_labels_u_init, digit_labels_f_next, digit_labels_r_next, digit_labels_u_next)
            sat_agg = SatAgg(
                *reconstruction_axioms_based_on_actions_1, *reconstruction_axioms_based_on_actions_2,
                *digit_classification_rules 
            )
        
        else:
            sat_agg = SatAgg(*reconstruction_axioms_based_on_actions_1, *reconstruction_axioms_based_on_actions_2)
    
        #print(f"sat agg: {sat_agg}")
        return sat_agg

    def compute_encoder_only_sat(self, init_image, actions_, next_image):
        # cube 1 -----------------------------------------------------------
        init_image_a_0 = ltn.Variable("init_image_a_0", init_image[actions_.squeeze(1) == 0]) if init_image[actions_.squeeze(1) == 0].shape[0] != 0 else None
        init_image_a_1 = ltn.Variable("init_image_a_1", init_image[actions_.squeeze(1) == 1]) if init_image[actions_.squeeze(1) == 1].shape[0] != 0 else None
        init_image_a_2 = ltn.Variable("init_image_a_2", init_image[actions_.squeeze(1) == 2]) if init_image[actions_.squeeze(1) == 2].shape[0] != 0 else None
        init_image_a_3 = ltn.Variable("init_image_a_3", init_image[actions_.squeeze(1) == 3]) if init_image[actions_.squeeze(1) == 3].shape[0] != 0 else None
        init_image_a_4 = ltn.Variable("init_image_a_4", init_image[actions_.squeeze(1) == 4]) if init_image[actions_.squeeze(1) == 4].shape[0] != 0 else None
        init_image_a_5 = ltn.Variable("init_image_a_5", init_image[actions_.squeeze(1) == 5]) if init_image[actions_.squeeze(1) == 5].shape[0] != 0 else None

        # cube 2 -----------------------------------------------------------
        next_image_a_0 = ltn.Variable("next_image_a_0", next_image[actions_.squeeze(1) == 0]) if next_image[actions_.squeeze(1) == 0].shape[0] != 0 else None
        next_image_a_1 = ltn.Variable("next_image_a_1", next_image[actions_.squeeze(1) == 1]) if next_image[actions_.squeeze(1) == 1].shape[0] != 0 else None
        next_image_a_2 = ltn.Variable("next_image_a_2", next_image[actions_.squeeze(1) == 2]) if next_image[actions_.squeeze(1) == 2].shape[0] != 0 else None
        next_image_a_3 = ltn.Variable("next_image_a_3", next_image[actions_.squeeze(1) == 3]) if next_image[actions_.squeeze(1) == 3].shape[0] != 0 else None
        next_image_a_4 = ltn.Variable("next_image_a_4", next_image[actions_.squeeze(1) == 4]) if next_image[actions_.squeeze(1) == 4].shape[0] != 0 else None
        next_image_a_5 = ltn.Variable("next_image_a_5", next_image[actions_.squeeze(1) == 5]) if next_image[actions_.squeeze(1) == 5].shape[0] != 0 else None

        actions_a_0 = ltn.Variable("actions_a_0", actions_[actions_.squeeze(1) == 0]) if actions_[actions_.squeeze(1) == 0].shape[0] != 0 else None
        actions_a_1 = ltn.Variable("actions_a_1", actions_[actions_.squeeze(1) == 1]) if actions_[actions_.squeeze(1) == 1].shape[0] != 0 else None
        actions_a_2 = ltn.Variable("actions_a_2", actions_[actions_.squeeze(1) == 2]) if actions_[actions_.squeeze(1) == 2].shape[0] != 0 else None
        actions_a_3 = ltn.Variable("actions_a_3", actions_[actions_.squeeze(1) == 3]) if actions_[actions_.squeeze(1) == 3].shape[0] != 0 else None
        actions_a_4 = ltn.Variable("actions_a_4", actions_[actions_.squeeze(1) == 4]) if actions_[actions_.squeeze(1) == 4].shape[0] != 0 else None
        actions_a_5 = ltn.Variable("actions_a_5", actions_[actions_.squeeze(1) == 5]) if actions_[actions_.squeeze(1) == 5].shape[0] != 0 else None
        reconstruction_axioms_based_on_actions_1 = self.get_encoder_rules(init_image_a_0, init_image_a_1, init_image_a_2, init_image_a_3, init_image_a_4, init_image_a_5, next_image_a_0, next_image_a_1, next_image_a_2, next_image_a_3, next_image_a_4, next_image_a_5, actions_a_0, actions_a_1, actions_a_2, actions_a_3, actions_a_4, actions_a_5)
        #reconstruction_axioms_based_on_actions_2 = self.get_decoder_rules(init_image_a_0, init_image_a_1, init_image_a_2, init_image_a_3, init_image_a_4, init_image_a_5, next_image_a_0, next_image_a_1, next_image_a_2, next_image_a_3, next_image_a_4, next_image_a_5, actions_a_0, actions_a_1, actions_a_2, actions_a_3, actions_a_4, actions_a_5)

        sat_agg = SatAgg(*reconstruction_axioms_based_on_actions_1)
    
        #print(f"sat agg: {sat_agg}")
        return sat_agg  

    def compute_decoder_only_sat(self, init_image, actions_, next_image):
        # cube 1 -----------------------------------------------------------
        init_image_a_0 = ltn.Variable("init_image_a_0", init_image[actions_.squeeze(1) == 0]) if init_image[actions_.squeeze(1) == 0].shape[0] != 0 else None
        init_image_a_1 = ltn.Variable("init_image_a_1", init_image[actions_.squeeze(1) == 1]) if init_image[actions_.squeeze(1) == 1].shape[0] != 0 else None
        init_image_a_2 = ltn.Variable("init_image_a_2", init_image[actions_.squeeze(1) == 2]) if init_image[actions_.squeeze(1) == 2].shape[0] != 0 else None
        init_image_a_3 = ltn.Variable("init_image_a_3", init_image[actions_.squeeze(1) == 3]) if init_image[actions_.squeeze(1) == 3].shape[0] != 0 else None
        init_image_a_4 = ltn.Variable("init_image_a_4", init_image[actions_.squeeze(1) == 4]) if init_image[actions_.squeeze(1) == 4].shape[0] != 0 else None
        init_image_a_5 = ltn.Variable("init_image_a_5", init_image[actions_.squeeze(1) == 5]) if init_image[actions_.squeeze(1) == 5].shape[0] != 0 else None

        # cube 2 -----------------------------------------------------------
        next_image_a_0 = ltn.Variable("next_image_a_0", next_image[actions_.squeeze(1) == 0]) if next_image[actions_.squeeze(1) == 0].shape[0] != 0 else None
        next_image_a_1 = ltn.Variable("next_image_a_1", next_image[actions_.squeeze(1) == 1]) if next_image[actions_.squeeze(1) == 1].shape[0] != 0 else None
        next_image_a_2 = ltn.Variable("next_image_a_2", next_image[actions_.squeeze(1) == 2]) if next_image[actions_.squeeze(1) == 2].shape[0] != 0 else None
        next_image_a_3 = ltn.Variable("next_image_a_3", next_image[actions_.squeeze(1) == 3]) if next_image[actions_.squeeze(1) == 3].shape[0] != 0 else None
        next_image_a_4 = ltn.Variable("next_image_a_4", next_image[actions_.squeeze(1) == 4]) if next_image[actions_.squeeze(1) == 4].shape[0] != 0 else None
        next_image_a_5 = ltn.Variable("next_image_a_5", next_image[actions_.squeeze(1) == 5]) if next_image[actions_.squeeze(1) == 5].shape[0] != 0 else None

        actions_a_0 = ltn.Variable("actions_a_0", actions_[actions_.squeeze(1) == 0]) if actions_[actions_.squeeze(1) == 0].shape[0] != 0 else None
        actions_a_1 = ltn.Variable("actions_a_1", actions_[actions_.squeeze(1) == 1]) if actions_[actions_.squeeze(1) == 1].shape[0] != 0 else None
        actions_a_2 = ltn.Variable("actions_a_2", actions_[actions_.squeeze(1) == 2]) if actions_[actions_.squeeze(1) == 2].shape[0] != 0 else None
        actions_a_3 = ltn.Variable("actions_a_3", actions_[actions_.squeeze(1) == 3]) if actions_[actions_.squeeze(1) == 3].shape[0] != 0 else None
        actions_a_4 = ltn.Variable("actions_a_4", actions_[actions_.squeeze(1) == 4]) if actions_[actions_.squeeze(1) == 4].shape[0] != 0 else None
        actions_a_5 = ltn.Variable("actions_a_5", actions_[actions_.squeeze(1) == 5]) if actions_[actions_.squeeze(1) == 5].shape[0] != 0 else None
        reconstruction_axioms_based_on_actions = self.get_decoder_rules(init_image_a_0, init_image_a_1, init_image_a_2, init_image_a_3, init_image_a_4, init_image_a_5, next_image_a_0, next_image_a_1, next_image_a_2, next_image_a_3, next_image_a_4, next_image_a_5, actions_a_0, actions_a_1, actions_a_2, actions_a_3, actions_a_4, actions_a_5)

        sat_agg = SatAgg(*reconstruction_axioms_based_on_actions)

        return sat_agg 

    def get_encoder_rules(self, init_image_a_0, init_image_a_1, init_image_a_2, init_image_a_3, init_image_a_4, init_image_a_5, next_image_a_0, next_image_a_1, next_image_a_2, next_image_a_3, next_image_a_4, next_image_a_5, actions_a_0, actions_a_1, actions_a_2, actions_a_3, actions_a_4, actions_a_5):
        rule_1 = self.encoder_constraints.EncoderRuleA0(init_image_a_0, next_image_a_0, actions_a_0) if init_image_a_0 is not None else None  
        rule_2 = self.encoder_constraints.EncoderRuleA1(init_image_a_1, next_image_a_1, actions_a_1) if init_image_a_1 is not None else None  
        rule_3 = self.encoder_constraints.EncoderRuleA2(init_image_a_2, next_image_a_2, actions_a_2) if init_image_a_2 is not None else None  
        rule_4 = self.encoder_constraints.EncoderRuleA3(init_image_a_3, next_image_a_3, actions_a_3) if init_image_a_3 is not None else None  
        rule_5 = self.encoder_constraints.EncoderRuleA4(init_image_a_4, next_image_a_4, actions_a_4) if init_image_a_4 is not None else None  
        rule_6 = self.encoder_constraints.EncoderRuleA5(init_image_a_5, next_image_a_5, actions_a_5) if init_image_a_5 is not None else None  

        rules = [rule_1, rule_2, rule_3, rule_4, rule_5, rule_6]
        rules = [rule for rule in rules if rule is not None]
        return rules

    def get_decoder_rules(self, init_image_a_0, init_image_a_1, init_image_a_2, init_image_a_3, init_image_a_4, init_image_a_5, next_image_a_0, next_image_a_1, next_image_a_2, next_image_a_3, next_image_a_4, next_image_a_5, actions_a_0, actions_a_1, actions_a_2, actions_a_3, actions_a_4, actions_a_5):
        rule_1 = self.decoder_constraints.DecoderRuleA0(init_image_a_0, next_image_a_0, actions_a_0) if init_image_a_0 is not None else None  
        rule_2 = self.decoder_constraints.DecoderRuleA1(init_image_a_1, next_image_a_1, actions_a_1) if init_image_a_1 is not None else None  
        rule_3 = self.decoder_constraints.DecoderRuleA2(init_image_a_2, next_image_a_2, actions_a_2) if init_image_a_2 is not None else None  
        rule_4 = self.decoder_constraints.DecoderRuleA3(init_image_a_3, next_image_a_3, actions_a_3) if init_image_a_3 is not None else None  
        rule_5 = self.decoder_constraints.DecoderRuleA4(init_image_a_4, next_image_a_4, actions_a_4) if init_image_a_4 is not None else None  
        rule_6 = self.decoder_constraints.DecoderRuleA5(init_image_a_5, next_image_a_5, actions_a_5) if init_image_a_5 is not None else None  

        rules = [rule_1, rule_2, rule_3, rule_4, rule_5, rule_6]
        rules = [rule for rule in rules if rule is not None]
        return rules
    
    def compute_equal_decoding_sat(self, pred, ground_truth):
        pred = ltn.Variable("pred", pred)
        ground_truth = ltn.Variable("ground_truth", ground_truth)
        sat_val = self.decoder_constraints.compute_equal_decoding_sat(pred, ground_truth)
        return sat_val

    def get_digit_rules(self, init_image, next_image, digit_labels_f_init, digit_labels_r_init, digit_labels_u_init, digit_labels_f_next, digit_labels_r_next, digit_labels_u_next):
        #same_digit_rules = self.digit_constraints.get_same_digit_rules(init_image_a_0, init_image_a_1, init_image_a_2, init_image_a_3, init_image_a_4, init_image_a_5, next_image_a_0, next_image_a_1, next_image_a_2, next_image_a_3, next_image_a_4, next_image_a_5)
        #different_digit_rules = self.digit_constraints.get_different_digit_rules(init_image_a_0, init_image_a_1, init_image_a_2, init_image_a_3, init_image_a_4, init_image_a_5, next_image_a_0, next_image_a_1, next_image_a_2, next_image_a_3, next_image_a_4, next_image_a_5)
        #general_digit_constraints = self.digit_constraints.general_digit_constraints(init_image_, next_image_)
        digit_rules = self.digit_constraints.get_digit_rules(init_image, next_image, digit_labels_f_init, digit_labels_r_init, digit_labels_u_init, digit_labels_f_next, digit_labels_r_next, digit_labels_u_next)

        #rules = get_digit_rules #[*same_digit_rules, *different_digit_rules, *general_digit_constraints]
        #rules = [rule for rule in rules if rule is not None]
        return digit_rules


 