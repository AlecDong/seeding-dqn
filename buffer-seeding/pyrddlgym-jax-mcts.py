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
import optuna
import tqdm

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

import shutil
def evaluate(self, env, episodes=1, verbose=False, render=False,seed=None):
    '''Custom eval function to evaluate non standard gym environments.
    '''
    
    # check compatibility with environment
    if env.vectorized != self.use_tensor_obs:
        raise ValueError(f'RDDLEnv vectorized flag must match use_tensor_obs '
                            f'of current policy, got {env.vectorized} and '
                            f'{self.use_tensor_obs}, respectively.')
    
    gamma = env.discount
    
    # get terminal width
    if verbose:
        width = shutil.get_terminal_size().columns
        sep_bar = '-' * width
    
    # start simulation
    history = np.zeros((episodes,))
    for episode in range(episodes):
        
        # restart episode
        total_reward, cuml_gamma = 0.0, 1.0
        self.reset()
        state, _ = env.reset(seed=seed)
        
        # printing
        if verbose:
            print(f'initial state = \n{self._format(state, width)}')
        
        # simulate to end of horizon
        for step in range(env.horizon):
            if render:
                env.render()
            
            # take a step in the environment
            action = self.sample_action(state)   
            simplified_action = rddl_to_gym_action(env, action)
            next_state, reward, terminated, truncated, _ = env.step(simplified_action)
            total_reward += reward * cuml_gamma
            cuml_gamma *= gamma
            done = terminated or truncated
            
            # printing
            if verbose: 
                print(f'{sep_bar}\n'
                    f'step   = {step}\n'
                    f'action = \n{self._format(action, width)}\n'
                    f'simplified_action = {simplified_action}\n'
                    f'state  = \n{self._format(next_state, width)}\n'
                    f'reward = {reward}\n'
                    f'done   = {done}', file=f)
            state = next_state
            if done:
                break
        
        if verbose:
            print(f'\n'
                    f'episode {episode + 1} ended with return {total_reward}\n'
                    f'{"=" * width}')
        history[episode] = total_reward
        
        # set the seed on the first episode only
        seed = None
    
    # summary statistics
    return {
        'mean': np.mean(history),
        'median': np.median(history),
        'min': np.min(history),
        'max': np.max(history),
        'std': np.std(history)
    }

def train_with_eval(problem, instance, rollout_horizon, delta, learning_rate, train_seconds, episodes=50):
    '''Train JaxMCTS with the given hyperparameters and evaluate its performance.'''
    env = pyRDDLGym.make(problem, instance,
                         base_class=SimplifiedActionRDDLEnv,
                         enforce_action_constraints=True)

    rddl = env.model

    agent = JaxMCTSController(
        JaxMCTSPlanner(
            rddl,
            rollout_horizon=rollout_horizon,
            delta=delta,
            optimizer=optax.rmsprop,
            optimizer_kwargs={'learning_rate': learning_rate}),
        train_seconds=train_seconds,
        key=random.PRNGKey(SEED)
    )

    stats = agent.evaluate(env, episodes=episodes, verbose=False, render=False)
    env.close()

    kwargs = {
        'rollout_horizon': rollout_horizon,
        'delta': delta,
        'learning_rate': learning_rate
    }

    with open(f"tuned_eval_stats_mcts.txt", "a+") as f:
        f.write(f"{problem} {instance}: {kwargs}\nStats: {stats}\n")

    return stats

def objective(trial, problem, instance):
    '''Objective function for Optuna to optimize JaxMCTS hyperparameters.'''
    rollout_horizon = trial.suggest_int("rollout_horizon", 1, 20)
    delta = trial.suggest_float("delta", 0.01, 1.0, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    train_seconds = 1

    eval_stats = train_with_eval(
        problem, instance, rollout_horizon, delta, learning_rate, train_seconds)

    return eval_stats['mean']

def optimize(problem, instance, n_trials=10):
    '''Optimize JaxMCTS hyperparameters using Optuna.'''
    study = optuna.create_study(direction="maximize")
    print(f"Optimizing hyperparameters for {problem}_{instance}")
    study.optimize(lambda trial: objective(trial, problem, instance), n_trials=n_trials)

    return study.best_value, study.best_params

if __name__ == '__main__':
    SEED=0
    np.random.seed(SEED)
    problems = [
        ('AcademicAdvising_MDP_ippc2014', "1"),
        ('Wildfire_MDP_ippc2014', "10"),
        ('AcademicAdvising_MDP_ippc2014', "7"),
        ('CooperativeRecon_MDP_ippc2011', "6"),
        ('MountainCar_Discrete_gym', "0"),
        ('CartPole_Discrete_gym', "0"),
        ('Blackjack_arcade', "0"),
        ('CooperativeRecon_MDP_ippc2011', "1"),
        ('CooperativeRecon_MDP_ippc2011', "3"),
        ('CrossingTraffic_MDP_ippc2014', "10"),
        ('CrossingTraffic_MDP_ippc2014', "4"),
        ('CrossingTraffic_MDP_ippc2014', "9"),
        ('Eight_arcade', "1"),
        # ('Sokoban_arcade', "0"), This one is not working
        ('TowerOfHanoi_arcade', "0"),
        ('Wildfire_MDP_ippc2014', "7"),
        ('Wildfire_MDP_ippc2014', "5"),
    ]

    best_hyperparams = [
        {'rollout_horizon': 19, 'delta': 0.8460299073458044, 'learning_rate': 0.0010094771306295388},
        {'rollout_horizon': 18, 'delta': 0.9309915251647889, 'learning_rate': 0.0004658120913181635},
        {'rollout_horizon': 18, 'delta': 0.6389564745281541, 'learning_rate': 0.006958734258513394},
        {'rollout_horizon': 5, 'delta': 0.12712743999832776, 'learning_rate': 0.0002446579953278301},
        {'rollout_horizon': 20, 'delta': 0.5523796063830402, 'learning_rate': 0.0006474485566805271},
        {'rollout_horizon': 13, 'delta': 0.2870366194508195, 'learning_rate': 0.0009526190874712655},
    ]
    
    for i, (problem, instance) in enumerate(problems):
        env = pyRDDLGym.make(problem, instance, 
                            base_class=SimplifiedActionRDDLEnv,
                            enforce_action_constraints=True)

        rddl = env.model

        rollout_horizon = best_hyperparams[i]['rollout_horizon']
        delta = best_hyperparams[i]['delta']
        learning_rate = best_hyperparams[i]['learning_rate']

        agent = JaxMCTSController(
            JaxMCTSPlanner(
                rddl, 
                rollout_horizon=rollout_horizon, 
                delta=delta,
                optimizer=optax.rmsprop,
                optimizer_kwargs={'learning_rate': learning_rate}),
            train_seconds=0.5,
            key=random.PRNGKey(SEED)
        )

        # monkey patch the evaluate function to use the custom one
        agent.evaluate = evaluate.__get__(agent, JaxMCTSController)
        all_stats=[]
        stats = agent.evaluate(env, episodes=50, verbose=False, render=False)
        all_stats.append(stats)
        print(all_stats)

        buffer_size = 100000
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
