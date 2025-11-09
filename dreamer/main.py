import argparse
from algorithm.dreamer import Dreamer
from dataset.dataset import Dataset
import ast

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--action_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--ep_len', type=int)
    parser.add_argument('--train_iterations', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--device', type=str)
    parser.add_argument('--login_key', type=str)
    parser.add_argument('--embed_state_size', type=int)
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--deter_size', type=int)
    parser.add_argument('--stoch_size', type=int)
    parser.add_argument('--obs_shape', type=str)
    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()

    datasetObject = Dataset(ast.literal_eval(args.obs_shape), args.action_size, args.device, args.dataset)
    dataset = datasetObject.get_dataset()

    dreamer = Dreamer(ast.literal_eval(args.obs_shape), args.stoch_size, args.deter_size, args.hidden_size, args.embed_state_size, args.action_size, dataset, args.batch_size, args.ep_len, args.train_iterations, args.lr, args.device, args.login_key)
    dreamer.train()



    


