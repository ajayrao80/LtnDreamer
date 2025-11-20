from ltn_model.ltn_qm.ltn_qm import LTNRules
from ltn_model.modules.model import LTNModel

class LogicLoss:
    def __init__(self, logic_model_path, model_name_front="front", model_name_right="right", model_name_up="up", model_name_dec="dec", model_name_rot_plus="rot_plus", model_name_rot_minus="rot_minus", model_name_digits="digits"):
        self.ltn_models = LTNModel()
        print(model_name_digits)
        self.ltn_models.load_all_models(logic_model_path, model_name_front, model_name_right, model_name_up, model_name_dec, model_name_rot_plus, model_name_rot_minus, model_name_digits)

        self.LTNObject = LTNRules(self.ltn_models)
    
    def compute_logic_loss(self, prev_state, action, next_state):
        sat_val = self.LTNObject.compute_sat(prev_state, action, next_state)
        loss = 1. - sat_val
        return loss








