# Logic Dreamer: Dataset Generation, Training, and Evaluation

Code (and dataset) for the paper "Integrating qualitative spatial model to reason about rotations into world models
using Logic Tensor Networks"

This repository provides a full pipeline for:
Dataset generation from environments

-Training Logic Dreamer (Dreamer + logic constraints)
-Training Vanilla Dreamer
-Evaluating both models
-Testing one-shot generalization using handwritten-letter environments

## 1. Dataset Generation

Generate episodic datasets from a given environment.
### Usage
```bash
python GenerateDataset.py \
  --env_path ENV_PATH_HERE \
  --path_to_save PATH_TO_SAVE_DATASET \
  --number_of_episodes NUMBER_OF_EPISODES \
  --episode_len EPISODE_LENGTH
```

### Arguments

```bash
--env_path: Path to the environment file
--path_to_save: Directory where the dataset will be saved
--number_of_episodes: Number of episodes to generate
--episode_len: Length of each episode
```

## 2. Logic Dreamer Training

### Train the Logic Dreamer using pre-generated datasets.

### Usage
```bash
python train_logic_dreamer.py \
  --train_dataset_path TRAIN_DATASET_PATH_HERE \
  --test_dataset_path TEST_DATASET_PATH_HERE \
  --model_save_path PATH_TO_SAVE_DREAMER_MODELS \
  --logic_models_path PATH_TO_SAVE_LOGIC_MODELS \
  --train_dataset_size TRAIN_DATASET_SIZE \
  --test_dataset_size TEST_DATASET_SIZE

### Optional Arguments
--episode_len
--batch_size
--project_name
--login_key
--free_nats
--logic_weight
--stoch_dim
--deter_dim
--lr
--epochs

### Default Values
stoch_dim     = 200
episode_len  = 5
batch_size   = 64
free_nats    = 3.0
logic_weight = 25000.0
lr           = 0.0001
epochs       = 1000
```

#### Note
```project_name``` and ```login_key``` are optional and only required if logging to Weights & Biases (wandb).

## 3. Vanilla Dreamer Training

### Train a standard Dreamer model without logic constraints.

### Usage
```bash
python train_vanilla_dreamer.py \
  --train_dataset_path TRAIN_DATASET_PATH_HERE \
  --test_dataset_path TEST_DATASET_PATH_HERE \
  --model_save_path DREAMER_MODEL_SAVE_PATH \
  --train_dataset_size TRAIN_DATASET_SIZE \
  --test_dataset_size TEST_DATASET_SIZE

### Optional Arguments
--lr
--epochs
--episode_len
--batch_size
--free_nats
--stoch_dim
--deter_dim

### Default Values
deter_dim    = 400
stoch_dim    = 200
embed_dim    = 200
episode_len = 5
batch_size  = 64
free_nats   = 3.0
lr          = 0.0001
epochs      = 1000
```

## 4. Logic Dreamer Evaluation

### Evaluate a trained Logic Dreamer model.

### Usage
```bash
python eval_logic_dreamer.py \
  --dataset_test_path DATASET_PATH_HERE \
  --logic_models_path LOGIC_MODELS_PATH_HERE \
  --world_model_upscale_network_name WORLD_MODEL_UPSCALE_NETWORK_NAME \
  --world_model_rssm_name WORLD_MODEL_RSSM_NAME \
  --ltn_front_name LTN_MODEL_FRONT_NAME \
  --ltn_right_name LTN_MODEL_RIGHT_NAME \
  --ltn_up_name LTN_MODEL_UP_NAME \
  --ltn_dec_name LTN_MODEL_DEC_NAME \
  --ltn_rot_change_name LTN_ROT_CHANGE_NAME

Optional Arguments
--deter_dim
--stoch_dim
--episode_len
--logic_models_path_best

Default Values
deter_dim    = 400
stoch_dim    = 200
episode_len = 5
logic_models_path_best = None
```

## 5. Vanilla Dreamer Evaluation

### Evaluate a trained Vanilla Dreamer model.

### Usage
```bash
python evaluate_vanilla_dreamer.py \
  --vanilla_model_path VANILLA_MODEL_PATH_HERE \
  --dataset_test_path TEST_DATASET_PATH_HERE \
  --logic_models_path LOGIC_MODELS_PATH_HERE \
  --world_model_upscale_network_name DREAMER_MODEL_UPSCALE_NETWORK_NAME \
  --world_model_encoder_name DREAMER_ENCODER_NAME \
  --world_model_decoder_name DREAMER_DECODER_NAME \
  --world_model_rssm_name DREAMER_RSSM_NAME

Optional Arguments
--embed_dim
--deter_dim
--stoch_dim
--episode_len
--logic_models_path_best

Default Values
deter_dim    = 400
stoch_dim    = 200
embed_dim    = 200
episode_len = 5
logic_models_path_best = None
```
## 6. Environments

### The environments are provided as a ZIP file and include:

#### Training Environment
Used to generate training datasets
#### Test Environment
Used to generate test datasets
#### Handwritten Letters Environment
Used to evaluate one-shot generalization on unseen handwritten symbols

### Recommended Workflow
1. Generate training and test datasets
2. Train Logic Dreamer and/or Vanilla Dreamer
3. Evaluate on standard test datasets
4. Evaluate on handwritten-letter datasets for one-shot generalization
