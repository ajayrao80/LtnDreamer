from dreamer.main import main
import argparse
import random
from dreamer.utils.utils import ReplayBuffer
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
import numpy as np

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
    
def create_dataset(train_dataset_size, episode_len, path_to_save, env_path):
    dataGen = DatasetGenerator(observation_shape=(3, 128, 128), action_size=7)
    unity_env = unity_env = UnityEnvironment(env_path, worker_id=0)  
    env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
    env.action_space.seed(45)
    dataset_train = dataGen.generate_dataset(env, train_dataset_size)
    env.close()
    dataset_train = dataset_train.sample(train_dataset_size, episode_len)
    observation_train = dataset_train.observation
    action_train = dataset_train.action
    obs_train = observation_train.reshape(observation_train.shape[0]*observation_train.shape[1], observation_train.shape[2], observation_train.shape[3], observation_train.shape[4])
    act_train = action_train.reshape(action_train.shape[0]*action_train.shape[1], action_train.shape[2])
    np.savez(path_to_save, observation=obs_train.cpu(), action=act_train.cpu())
    return path_to_save

def train_ltn_world_model(lr, epochs, login_key, model_save_path, train_dataset_path, test_dataset_path, free_nats, stoch_dim, deter_dim, embed_dim, beta=1.0, project_name=None, train_dataset_size=10, test_dataset_size=10, episode_len=5, batch_size=32):
    lr = lr
    epochs = epochs
    embed_dim = embed_dim
    stoch_dim = stoch_dim
    deter_dim = deter_dim
    beta = beta
    project_name = project_name
    login_key = login_key
    model_save_path = model_save_path
    train_dataset_size = train_dataset_size
    test_dataset_size = test_dataset_size
    episode_len = episode_len
    batch_size = batch_size
    #train_env_path = train_env_path
    #test_env_path = test_env_path
    free_nats = free_nats
    
    #train_dataset_path = create_dataset(train_dataset_size, episode_len, "./world_model_train.npz", train_env_path)
    #test_dataset_path = create_dataset(test_dataset_size, episode_len, "./world_model_test.npz", test_env_path)
    main(lr, epochs, embed_dim, stoch_dim, deter_dim, train_dataset_path, test_dataset_path, beta, login_key, model_save_path, free_nats=free_nats, project_name=project_name, batch_size=batch_size, episode_len=episode_len)     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--login_key', type=str, default=None)
    parser.add_argument('--model_save_path', type=str, default="../")
    parser.add_argument('--train_dataset_path', type=str, required=True)
    parser.add_argument('--test_dataset_path', type=str, required=True)
    parser.add_argument('--project_name', type=str, default=None)
    parser.add_argument('--train_dataset_size', type=int, default=5000)
    parser.add_argument('--test_dataset_size', type=int, default=1000)
    parser.add_argument('--episode_len', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--free_nats', type=float, default=3.0)
    parser.add_argument('--stoch_dim', type=int, default=200)
    parser.add_argument('--deter_dim', type=int, default=400)
    parser.add_argument('--embed_dim', type=int, default=200)
    args = parser.parse_args()
    train_ltn_world_model(args.lr, args.epochs, args.login_key, args.model_save_path, args.train_dataset_path, args.test_dataset_path, args.free_nats, args.stoch_dim, args.deter_dim, args.embed_dim, project_name=args.project_name, train_dataset_size=args.train_dataset_size, test_dataset_size=args.test_dataset_size, episode_len=args.episode_len, batch_size=args.batch_size)