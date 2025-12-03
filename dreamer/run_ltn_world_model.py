"""
from dreamer.ltn_world_model import main
lr = 0.0001
epochs = 1
embed_dim = 128*14*14
dataset_train_path = "../EpisodeTrajectoryDataset/world_model_train.npz"
dataset_test_path = "../EpisodeTrajectoryDataset/world_model_test.npz"
project_name = "logic_injected_world_model"
login_key = "617b36c0788d486c8c687bfbc3e573b779f42c0a"
model_save_path = "../dreamer_models/ltn"
logic_models_path = "../ltn_models"
main(lr, epochs, embed_dim, dataset_train_path, dataset_test_path, login_key, model_save_path, logic_models_path=logic_models_path, project_name=project_name, train_all=True, batch_size=128)
"""

from dreamer.ltn_world_model import main
lr = 0.0001
epochs = 100
embed_dim = 128*14*14
project_name = "logic_injected_world_model"
login_key = "617b36c0788d486c8c687bfbc3e573b779f42c0a"
model_save_path = "../dreamer_models/ltn"
logic_models_path = "../ltn_models"
dataset_train_path = "../world_model_test.npz"
dataset_test_path = "../world_model_test.npz"
main(lr, epochs, embed_dim, dataset_train_path, dataset_test_path, login_key, model_save_path, logic_models_path=logic_models_path, project_name=project_name, train_all=True, batch_size=1)
