from utils.utils import ReplayBuffer
import numpy as np 

class Dataset:
    def __init__(self, observation_shape, action_size, _device, _dataset_path, _dataset_path_test):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.device = _device
        self.dataset_path = _dataset_path
        self.dataset_path_test = _dataset_path_test

    def get_dataset_train(self):
        loaded_data = np.load(self.dataset_path)
        buffer = ReplayBuffer(self.observation_shape, self.action_size, self.device, capacity=loaded_data["observation"].shape[0])
        for observation, action in zip(loaded_data["observation"], loaded_data["action"]):
            buffer.add(observation=observation, action=action)
        
        return buffer
    
    def get_dataset_test(self):
        loaded_data = np.load(self.dataset_path_test)
        buffer = ReplayBuffer(self.observation_shape, self.action_size, self.device, capacity=loaded_data["observation"].shape[0])
        for observation, action in zip(loaded_data["observation"], loaded_data["action"]):
            buffer.add(observation=observation, action=action)
        
        return buffer
    
