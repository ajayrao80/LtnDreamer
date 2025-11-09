from utils.buffer import ReplayBuffer
import numpy as np

class Dataset:
    def __init__(self, observation_shape, action_size, _device, _dataset_path):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.device = _device
        self.dataset_path = _dataset_path

    def get_dataset(self):
        loaded_data = np.load(self.dataset_path)
        buffer = ReplayBuffer(self.observation_shape, self.action_size, self.device)
        for observation, action, next_observation in zip(loaded_data["observation"], loaded_data["action"], loaded_data["next_observation"]):
            buffer.add(observation=observation, action=action, next_observation=next_observation)
        
        return buffer



