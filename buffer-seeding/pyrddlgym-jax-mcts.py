from pyRDDLGym_rl.core.env import SimplifiedActionRDDLEnv
import pyRDDLGym
# For now, need to checkout the Jax-MCTS-branch branch
from pyRDDLGym_jax.core.mcts import JaxMCTSController, JaxMCTSPlanner
from stable_baselines3.common.buffers import DictReplayBuffer
import os
import pickle
import numpy as np
import optax
from jax import random


def rddl_to_gym_action(env, action):
    # Turn action into one processable by the simplified environment
    locational, _, bool_constraint, _ = env._action_info

    if bool_constraint:
        # this should be empty action
        act = env.action_space.n - 1

        for key in locational:
            true_loc = np.where(action[key])[0]
            if true_loc.size > 0:
                _, (start, _) = locational[key]
                act = start + true_loc
    else:
        act = list(action.values())[0]
    
    return act


if __name__ == '__main__':
    SEED=0
    np.random.seed(SEED)

    problems = [
        ('MountainCar_Discrete_gym', "0"),
        ('CartPole_Discrete_gym', "0"),
        ('Blackjack_arcade', "0"),
        ('CooperativeRecon_MDP_ippc2011', "1"),
        ('CooperativeRecon_MDP_ippc2011', "3"),
        ('CooperativeRecon_MDP_ippc2011', "6"),
        ('CrossingTraffic_MDP_ippc2014', "10"),
        ('CrossingTraffic_MDP_ippc2014', "4"),
        ('CrossingTraffic_MDP_ippc2014', "9"),
        ('Eight_arcade', "1"),
        # ('Sokoban_arcade', "0"), This one is not working
        ('TowerOfHanoi_arcade', "0"),
        ('Wildfire_MDP_ippc2014', "10"),
        ('Wildfire_MDP_ippc2014', "7"),
        ('Wildfire_MDP_ippc2014', "5"),
        ('AcademicAdvising_MDP_ippc2014', "1"),
        ('AcademicAdvising_MDP_ippc2014', "7")
    ]
    for problem, instance in problems:
        env = pyRDDLGym.make(problem, instance, 
                            base_class=SimplifiedActionRDDLEnv,
                            enforce_action_constraints=True)

        rddl = env.model

        agent = JaxMCTSController(
            JaxMCTSPlanner(
                rddl, 
                rollout_horizon=5, 
                delta=0.2,
                optimizer=optax.rmsprop,
                optimizer_kwargs={'learning_rate': 0.003}),
            train_seconds=10,
            key=random.PRNGKey(SEED)
        )

        buffer_size = 10000
        replay_buffer = DictReplayBuffer(
            buffer_size,
            env.observation_space,
            env.action_space,
            device="cpu",
            n_envs=1,
            optimize_memory_usage=False,
            handle_timeout_termination=True
        )

        while not replay_buffer.full:
            state, _ = env.reset()
            done = False
            while not done:
                # Get action
                action = agent.sample_action(state)
                simplified_action = rddl_to_gym_action(env, action)

                # Take step
                next_state, reward, terminated, truncated, info = env.step(simplified_action)
                done = terminated or truncated or replay_buffer.full

                # Add transition to replay buffer
                replay_buffer.add(
                    state, 
                    next_state,
                    simplified_action,
                    reward,
                    done,
                    [info]
                )

                state = next_state

        env.close()
        save_dir = "replay_buffer_mcts"
        os.makedirs(save_dir, exist_ok=True)

        with open(f"{save_dir}/replay_buffer_{problem}_{instance}.pkl", "wb") as f:
            pickle.dump(replay_buffer, f)

        # Save a human-readable summary
        with open(os.path.join(save_dir, f"{problem}_{instance}_buffer_summary.txt"), "w") as f:
            f.write(f"Replay Buffer Summary\n")
            f.write(f"====================\n")
            f.write(f"Total number of transitions: {replay_buffer.size()}\n")
            f.write(f"Observation space: {env.observation_space}\n")
            f.write(f"Action space: {env.action_space}\n")
            f.write(f"Buffer position: {replay_buffer.pos}\n")
            f.write(f"Buffer full: {replay_buffer.full}\n")

        print(f"Buffer size: {replay_buffer.size()}")
        print(f"Data saved to directory: {save_dir}")
