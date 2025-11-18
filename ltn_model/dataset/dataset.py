import torch
import numpy as np

class CubeDataset(torch.utils.data.Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset["initial_state_img"][index], self.dataset["numbers_initial"][index], self.dataset["action"][index], self.dataset["next_state_img"][index], self.dataset["numbers_next"][index]
    
def load_dataset(path):
    dataset = np.load(path, allow_pickle=True)
    return dataset

def get_dataset(dataset_path):
    dst = load_dataset(dataset_path)
    dataset = CubeDataset(dst)
    return dataset

