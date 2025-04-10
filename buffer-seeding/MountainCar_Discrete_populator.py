import pyRDDLGym
from pyRDDLGym_jax.core.planner import JaxBackpropPlanner, JaxOfflineController, load_config
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
from gymnasium import spaces
import os
import pickle
from tkinter import Tk, filedialog

def shuffle_replay_buffer(buffer):
    """Shuffle all transitions in the replay buffer while maintaining relationships."""
    size = buffer.size()
    if size == 0:
        return
    
    # Create a random permutation of indices
    indices = np.random.permutation(size)
    
    # Shuffle each component using the same indices
    buffer.observations[:size] = buffer.observations[indices]
    buffer.next_observations[:size] = buffer.next_observations[indices]
    buffer.actions[:size] = buffer.actions[indices]
    buffer.rewards[:size] = buffer.rewards[indices]
    buffer.dones[:size] = buffer.dones[indices]
    buffer.timeouts[:size] = buffer.timeouts[indices]

# Mountain Car Discrete Gym Environment Setup
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select the .cfg file", filetypes=[("Config files", "*.cfg")])
# Set up the environment
env = pyRDDLGym.make("MountainCar_Discrete_gym", "0", vectorized=True)
planner_args, _, train_args = load_config(file_path)

# Create the planning algorithm
planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
controller = JaxOfflineController(planner, **train_args)

# Create a replay buffer
# Note: You'll need to specify the correct observation and action spaces
# This is an example - adjust the sizes based on your environment
obs_dim = len(env.observation_space.sample())  # Get observation dimension
action_dim = len(env.action_space.sample())    # Get action dimension
buffer_size = 20000000  # Adjust based on your needs

# Create continuous action space between 0 and 2
# We need to extract the actual Box space from the dictionary action space
action_space = env.action_space['action']  # This should be the Box(0.0, 2.0, (1,), float32)

# Create a concatenated observation space
# For MountainCar, we have 2 observations: position and velocity
observation_space = spaces.Box(
    low=np.array([-1.2, -0.07]),  # Concatenated lower bounds
    high=np.array([0.6, 0.07]),   # Concatenated upper bounds
    shape=(2,),                    # Total number of observations
    dtype=np.float64
)

replay_buffer = ReplayBuffer(
    buffer_size,
    observation_space,  # Use our concatenated observation space
    action_space,
    device="auto",
    n_envs=1,
    optimize_memory_usage=False,
    handle_timeout_termination=True
)

for episode in range(1000):
    # Evaluate the planner
    if episode % 50 == 0:
        print(f"Episode {episode}")
    state, _ = env.reset()
    done = False

    while not done:
        # Get action
        action = controller.sample_action(state)
        
        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Add transition to replay buffer
        # Convert state dictionary to concatenated array with correct shape
        obs = np.array([state['pos'], state['vel']], dtype=np.float64).reshape(1, -1)  # Shape: (1,2)
        next_obs = np.array([next_state['pos'], next_state['vel']], dtype=np.float64).reshape(1, -1)  # Shape: (1,2)
        act = action['action']
        
        # Convert single info dict to list of info dicts
        infos = [info]
        
        replay_buffer.add(
            obs,
            next_obs,
            act,
            reward,
            done,
            infos  # Pass as a list of info dicts
        )
        
        state = next_state

env.close()

# Shuffle the replay buffer before saving
print("Shuffling replay buffer...")
shuffle_replay_buffer(replay_buffer)

# Create a directory to save the data if it doesn't exist
save_dir = "replay_buffer_data"
os.makedirs(save_dir, exist_ok=True)

# Save the replay buffer data in a format compatible with DQN's load_replay_buffer
buffer_data = {
    'observations': replay_buffer.observations[:replay_buffer.size()],
    'next_observations': replay_buffer.next_observations[:replay_buffer.size()],
    'actions': replay_buffer.actions[:replay_buffer.size()],
    'rewards': replay_buffer.rewards[:replay_buffer.size()],
    'dones': replay_buffer.dones[:replay_buffer.size()],
    'timeouts': replay_buffer.timeouts[:replay_buffer.size()],
    'pos': replay_buffer.pos,
    'full': replay_buffer.full,
    'observation_space': observation_space,
    'action_space': action_space,
    'buffer_size': buffer_size,
    'n_envs': 1,
    'optimize_memory_usage': False,
    'handle_timeout_termination': True
}

# Save the buffer data
with open(os.path.join(save_dir, "replay_buffer.pkl"), "wb") as f:
    pickle.dump(buffer_data, f)

# Save a human-readable summary
with open(os.path.join(save_dir, "buffer_summary.txt"), "w") as f:
    f.write(f"Replay Buffer Summary\n")
    f.write(f"====================\n")
    f.write(f"Total number of transitions: {replay_buffer.size()}\n")
    f.write(f"Observation space: {observation_space}\n")
    f.write(f"Action space: {action_space}\n")
    f.write(f"Buffer position: {replay_buffer.pos}\n")
    f.write(f"Buffer full: {replay_buffer.full}\n")

print(f"Buffer size: {replay_buffer.size()}")
print(f"Data saved to directory: {save_dir}")

# Example of how to use the saved buffer with DQN:
"""
from stable_baselines3 import DQN

# Create a DQN model with the same observation and action spaces
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.0001,
    buffer_size=buffer_size,
    learning_starts=1000,
    batch_size=64,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    max_grad_norm=10,
    stats_window_size=100,
    tensorboard_log="./dqn_tensorboard/",
    create_eval_env=False,
    policy_kwargs=None,
    verbose=1,
    seed=None,
    device='auto',
    _init_setup_model=True
)

# Load the replay buffer
model.load_replay_buffer("replay_buffer_data/replay_buffer.pkl")
"""
