import torch
import numpy as np

class CubeDataset(torch.utils.data.Dataset):
    def __init__(self, _dataset):
        self.initial_state_image = _dataset["initial_state_img"]
        self.numbers_initial = _dataset["numbers_initial"]
        self.action = _dataset["action"]
        self.next_state_img = _dataset["next_state_img"]
        self.numbers_next = _dataset["numbers_next"]
    
    def __len__(self):
        return len(self.initial_state_image)
    
    def __getitem__(self, index):
        return self.initial_state_image[index], self.numbers_initial[index], self.action[index], self.next_state_img[index], self.numbers_next[index] 
    
def load_dataset(path):
    dataset = np.load(path, allow_pickle=True)
    return dataset

def get_dataset(dataset_path):
    dst = load_dataset(dataset_path)
    dataset = CubeDataset(dst)
    return dataset

