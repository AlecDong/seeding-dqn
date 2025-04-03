import optuna
from stable_baselines3 import DQN
import tqdm

import pyRDDLGym
from pyRDDLGym_rl.core.agent import StableBaselinesAgent
from pyRDDLGym_rl.core.env import SimplifiedActionRDDLEnv

SEED=0

def train(problem, instance, lr, batch_size, eps_final, gamma, buffer_size, 
          target_update_freq, train_freq, gradient_steps, steps=200000, 
          load_replay_buffer=False):
    # set up the environment
    env = pyRDDLGym.make(problem, instance, 
                         base_class=SimplifiedActionRDDLEnv,
                         enforce_action_constraints=True)
    kwargs = {
        'verbose': 0, 
        'device': 'cpu', 
        'seed': SEED, 
        '_init_setup_model': True,
        'learning_rate': lr,
        'batch_size': batch_size,
        'exploration_final_eps': eps_final,
        'gamma': gamma,
        'buffer_size': buffer_size,
        'target_update_interval': target_update_freq,
        'train_freq': train_freq,
        'gradient_steps': gradient_steps
    }
    model = DQN('MultiInputPolicy', env, **kwargs)

    if load_replay_buffer:
        print("Loading replay buffer from file")
        model.load_replay_buffer(f"replay_buffer_mcts/replay_buffer_{problem}_{instance}.pkl")

    model.learn(total_timesteps=steps)

    return env, model

def train_with_eval_every_n_steps(problem, instance, lr, batch_size, eps_final, gamma, buffer_size, 
          target_update_freq, train_freq, gradient_steps, steps=200000, 
          load_replay_buffer=False, n_steps=1000):
    env = pyRDDLGym.make(problem, instance, 
                         base_class=SimplifiedActionRDDLEnv,
                         enforce_action_constraints=True)
    kwargs = {
        'verbose': 0, 
        'device': 'cpu', 
        'seed': SEED, 
        '_init_setup_model': True,
        'learning_rate': lr,
        'batch_size': batch_size,
        'exploration_final_eps': eps_final,
        'gamma': gamma,
        'buffer_size': buffer_size,
        'target_update_interval': target_update_freq,
        'train_freq': train_freq,
        'gradient_steps': gradient_steps
    }
    model = DQN('MultiInputPolicy', env, **kwargs)

    if load_replay_buffer:
        print("Loading replay buffer from file")
        model.load_replay_buffer(f"replay_buffer_mcts/replay_buffer_{problem}_{instance}.pkl")

    evals = []
    for step in tqdm.tqdm(range(0, steps, n_steps)):
        model.learn(total_timesteps=n_steps)
        print(f"Evaluating model after {step + n_steps} steps")
        evals.append(evaluate(env, model))
    
    return evals


def evaluate(env, model):
    dqn_agent = StableBaselinesAgent(model)
    eval_stats = dqn_agent.evaluate(env, episodes=5, verbose=False, render=False, seed=100)

    return eval_stats

def objective(trial, problem, instance, steps=200000, load_replay_buffer=False):
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    eps_final = trial.suggest_float("eps_final", 0.01, 0.1, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    buffer_size = trial.suggest_categorical("buffer_size", [10000, 50000, 100000])
    target_update_freq = trial.suggest_categorical("target_update_freq", [100, 1000])
    train_freq = trial.suggest_categorical("train_freq", [4, 16, 64])
    gradient_steps = trial.suggest_categorical("gradient_steps", [1, 8, 16])
    env, model = train(problem, instance, lr, batch_size, eps_final, gamma, 
                       buffer_size, target_update_freq, train_freq, 
                       gradient_steps, steps, load_replay_buffer)

    eval_stats = evaluate(env, model)

    env.close()

    return eval_stats['mean']

def optimize(problem, instance, steps=200000):
    study = optuna.create_study(direction="maximize")
    print(f"Optimizing hyperparameters for {problem}_{instance} ({steps} steps)")
    study.optimize(lambda trial: objective(trial, problem, instance, steps), n_trials=50)

    return study.best_value, study.best_params

def main():
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
        ('Sokoban_arcade', "0"),
        ('TowerOfHanoi_arcade', "0"),
        ('Wildfire_MDP_ippc2014', "10"),
        ('Wildfire_MDP_ippc2014', "7"),
        ('Wildfire_MDP_ippc2014', "5"),
        ('AcademicAdvising_MDP_ippc2014', "1"),
        ('AcademicAdvising_MDP_ippc2014', "7")
    ]

    for problem, instance in tqdm.tqdm(problems):
        best_value, best_params = optimize(problem, instance)
        with open(f"best_hyperparams.txt", "a+") as f:
            f.write(f"Problem: {problem}, Instance: {instance}\n")
            f.write(f"Best value obtained: {best_value} with parameters: {best_params}\n")
        print(f"Best value for {problem}_{instance}: {best_value} with params: {best_params}")
        print(f"Best value obtained: {best_value} with parameters: {best_params}")

if __name__ == "__main__":
    main()
