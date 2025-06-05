from common.double_replay_buffer import DoubleReplayBufferDQN
import optuna
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os

import pyRDDLGym
from pyRDDLGym_rl.core.agent import StableBaselinesAgent
from pyRDDLGym_rl.core.env import SimplifiedActionRDDLEnv

from stable_baselines3.common.callbacks import BaseCallback

SEED=0

class CustomEvalAndPlotCallback(BaseCallback):
    def __init__(self, env, evaluate_fn, plot_fn, plot_path, n_steps=10000, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.evaluate_fn = evaluate_fn
        self.plot_fn = plot_fn
        self.eval_interval = n_steps
        self.evals = []
        self.max_mean = float('-inf')
        self.best_eval_stats = {}
        self.plot_path = plot_path

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_interval == 0:
            eval_stats = self.evaluate_fn(self.env, self.model)
            self.evals.append(eval_stats)

            mean = eval_stats['mean']
            if mean > self.max_mean:
                self.max_mean = mean
                self.best_eval_stats = eval_stats

            self.plot_fn(self.evals, self.plot_path, n_steps=self.eval_interval)

        return True

def plot(data, filename, n_steps=10000):
    df = pd.DataFrame(data)
    x = df.index * n_steps
    y = df['mean']

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    # restart the plot
    plt.clf()
    plt.plot(x, y, color='blue')
    plt.xlabel('Steps')
    plt.ylabel('Mean')
    plt.title('Mean Reward')
    plt.grid()
    plt.savefig(filename, dpi=1200, bbox_inches='tight')


def train_with_eval_every_n_steps(problem, instance, lr, batch_size, eps_final, gamma, buffer_size, 
          target_update_freq, train_freq, gradient_steps, max_grad_norm, steps=1000000, 
          load_replay_buffer=True, n_steps=10000, secondary_frac=0.5):
    env = pyRDDLGym.make(problem, instance, 
                         base_class=SimplifiedActionRDDLEnv,
                         enforce_action_constraints=True)
    
    eval_env = pyRDDLGym.make(problem, instance, 
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
        'gradient_steps': gradient_steps,
        "max_grad_norm": max_grad_norm,
    }
    model = DoubleReplayBufferDQN(100000, secondary_frac, 'MultiInputPolicy', env, **kwargs)

    if load_replay_buffer:
        print("Loading replay buffer from file")
        model.load_replay_buffer(
            f"replay_buffer_mcts/replay_buffer_{problem}_{instance}.pkl", 
            "secondary"
        )
        
    print("Loaded replay buffer")

    # Callback for custom evaluation & plotting
    callback = CustomEvalAndPlotCallback(
        eval_env,
        evaluate_fn=evaluate,
        plot_fn=plot,
        plot_path=f"plots_double_replay_buffer/{problem}_{instance}_{secondary_frac}/lr_{lr}_eps_final_{eps_final}.png",
        n_steps=n_steps
    )

    model.learn(total_timesteps=steps, callback=callback)

    env.close()

    with open(f"tuned_eval_stats_double_replay.txt", "a+") as f:
        f.write(f"{problem} {instance}, Frac: {secondary_frac}: {kwargs}\nBest stats: {callback.best_eval_stats}\n")

    return callback.best_eval_stats

def evaluate(env, model):
    dqn_agent = StableBaselinesAgent(model)
    eval_stats = dqn_agent.evaluate(env, episodes=50, verbose=False, render=False, seed=100)

    return eval_stats


def objective(trial, problem, instance, steps=1000000, load_replay_buffer=True, secondary_frac=0.5):
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    eps_final = trial.suggest_float("eps_final", 0.01, 0.1, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    buffer_size = trial.suggest_categorical("buffer_size", [50000, 100000, 500000, 1000000, 5000000])
    target_update_freq = trial.suggest_categorical("target_update_freq", [100, 1000])
    train_freq = trial.suggest_categorical("train_freq", [4, 16, 64])
    gradient_steps = trial.suggest_categorical("gradient_steps", [1, 8, 16])
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 2.0)

    eval_stats = train_with_eval_every_n_steps(
        problem, instance, lr, batch_size, eps_final, gamma, 
        buffer_size, target_update_freq, train_freq, 
        gradient_steps, max_grad_norm, steps, load_replay_buffer, 10000, 
        secondary_frac)

    return eval_stats['mean']


def optimize(problem, instance, steps=1000000, secondary_frac=0.5):
    study = optuna.create_study(direction="maximize")
    print(f"Optimizing hyperparameters for {problem}_{instance} ({steps} steps)")
    print(f"Secondary fraction: {secondary_frac}")
    study.optimize(lambda trial: objective(trial, problem, instance, steps, True, secondary_frac), n_trials=50)

    return study.best_value, study.best_params


def main():
    problems = [
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
        ('AcademicAdvising_MDP_ippc2014', "7"),
        ('Wildfire_MDP_ippc2014', "10"),
        ('Wildfire_MDP_ippc2014', "7"),
        ('Wildfire_MDP_ippc2014', "5"),
        ('AcademicAdvising_MDP_ippc2014', "1"),
        ('MountainCar_Discrete_gym', "0"),
        ('CartPole_Discrete_gym', "0")
    ]

    fracs = [0.6, 0.4, 0.2]
    for frac in fracs:
        for problem, instance in tqdm.tqdm(problems):
            best_value, best_params = optimize(problem, instance, secondary_frac=frac)
            with open(f"best_hyperparams_double_replay.txt", "a+") as f:
                f.write(f"Problem: {problem}, Instance: {instance}, Frac: {frac}\n")
                f.write(f"Best value obtained: {best_value} with parameters: {best_params}\n")
            print(f"Best value for {problem}_{instance}: {best_value} with params: {best_params}")


if __name__ == "__main__":
    main()
