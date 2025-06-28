# Seeding-DQN
This project aims to bootstrap DQN with a pre-filled experience replay buffer to help guide exploration.
This builds off of [pyRDDLGym](https://github.com/pyrddlgym-project) and [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).

## Setup
Set up your Python environment and install the required packages with 
```
pip install -r requirements.txt
```
See https://pytorch.org/get-started/locally/ for getting torch to work with gpu

## Running
Running the baseline with Optuna hyperparameter tuning: `python -m dqn-training.stable_baselines_optuna`

Generating replay buffers with mcts: `python -m buffer-seeding.pyrddlgym-jax-mcts`
These buffers will be stored in pkl files like `replay_buffer_mcts/replay_buffer_{problem}_{instance}.pkl`

Using the generated replay buffers with DQN and Optuna hyperparameter tuning: `python -m dqn-training.double_replay_buffer_dqn_optuna`
