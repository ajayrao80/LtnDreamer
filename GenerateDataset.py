from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
import numpy as np
from PIL import Image
import random
import copy
from dreamer.utils.utils import ReplayBuffer

class DatasetGenerator:
    def __init__(self, observation_shape=(3, 128, 128), action_size=7):
        self.observation_shape = observation_shape
        self.action_size = action_size
    
    def environment_interaction(self, env, num_interaction_episodes, observation_shape, action_size, capacity, ep_len=5):
        buffer = ReplayBuffer(observation_shape, action_size, capacity=capacity)
        for epi in range(num_interaction_episodes):
            action = [-2, -2, -2, -2, -1, -2, -2] 

            observation = env.reset()[0]
        
            done = False

            steps = 0

            while not done:
                action_idx = random.randrange(0, 6)
                action = [-1, -1, -1, -1, action_idx, -1, -1]

                next_state, _, done, _ = env.step(action)

                buffer.add(
                    observation, np.array(action) #, next_state[0]
                )

                observation = next_state[0]

                steps += 1

                if steps >= ep_len:
                    observation = env.reset()[0]
                    done = True
        
        return buffer
    
    def generate_dataset(self, env, num_episodes, episode_len=5):
        c = num_episodes*episode_len
        dataset = self.environment_interaction(env, num_episodes, self.observation_shape, self.action_size, capacity=c, ep_len=episode_len)
        return dataset

def create_dataset(dataset_size, path_to_save, env_path, episode_len=5):
    dataGen = DatasetGenerator(observation_shape=(3, 128, 128), action_size=7)
    unity_env = UnityEnvironment(env_path, worker_id=0)  
    env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
    env.action_space.seed(45)
    dataset_train = dataGen.generate_dataset(env, dataset_size, episode_len=episode_len)
    env.close()
    dataset_train = dataset_train.sample(dataset_size, episode_len)
    observation_train = dataset_train.observation
    action_train = dataset_train.action
    obs_train = observation_train.reshape(observation_train.shape[0]*observation_train.shape[1], observation_train.shape[2], observation_train.shape[3], observation_train.shape[4])
    act_train = action_train.reshape(action_train.shape[0]*action_train.shape[1], action_train.shape[2])
    np.savez_compressed(path_to_save, observation=obs_train.cpu().numpy(), action=act_train.cpu().numpy())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--number_of_episodes', type=int, required=True)
    parser.add_argument('--episode_len', type=int, default=5)
    parser.add_argument('--path_to_save', type=str, required=True)
    parser.add_argument('--env_path', type=str, required=True)
    args = parser.parse_args()
    create_dataset(args.number_of_episodes, args.path_to_save, args.env_path, args.episode_len)
    print("Dataset created and saved to ", args.path_to_save)