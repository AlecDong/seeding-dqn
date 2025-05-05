from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples, DictReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.save_util import load_from_pkl

import numpy as np
import psutil
import warnings
from typing import Any, Optional, Union
import torch as th
from gymnasium import spaces
import io
import pathlib

from common.double_dqn import DoubleDQN

class DoubleReplayBuffer:
    """
    This is a wrapper around two replay buffers.
    
    Parameters:
        total_buffer_size (int): The maximum size of the replay buffer.
        secondary_buffer_fraction (float): The size of the secondary buffer as 
            a fraction of the total buffer size.
        observation_space (gym.Space): The observation space of the environment.
        action_space (gym.Space): The action space of the environment.
        device (str): The device to store the buffer on (e.g., 'cpu' or 'cuda').
        n_envs (int): The number of environments.
        optimize_memory_usage (bool): Whether to optimize memory usage.
        handle_timeout_termination (bool): Whether to handle timeout termination.
    """
    def __init__(
        self,
        total_buffer_size: int,
        secondary_buffer_fraction: float,
        replay_buffer_class: type[ReplayBuffer],
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        secondary_buffer_size = int(total_buffer_size * secondary_buffer_fraction)
        primary_buffer_size = total_buffer_size - secondary_buffer_size
        
        self.primary_buffer = replay_buffer_class(
            primary_buffer_size, 
            observation_space, 
            action_space, 
            device, 
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )

        self.secondary_buffer = replay_buffer_class(
            secondary_buffer_size, 
            observation_space, 
            action_space, 
            device, 
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )

        if psutil is not None:
            mem_available = psutil.virtual_memory().available

            obs_nbytes = 0
            for _, obs in self.primary_buffer.observations.items():
                obs_nbytes += obs.nbytes
            for _, obs in self.secondary_buffer.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage = (
                obs_nbytes + 
                self.primary_buffer.actions.nbytes +
                self.primary_buffer.rewards.nbytes + 
                self.primary_buffer.dones.nbytes +
                self.secondary_buffer.actions.nbytes +
                self.secondary_buffer.rewards.nbytes + 
                self.secondary_buffer.dones.nbytes
            )

            if not optimize_memory_usage:
                for _, obs in self.primary_buffer.next_observations.items():
                    total_memory_usage += obs.nbytes
                for _, obs in self.secondary_buffer.next_observations.items():
                    total_memory_usage += obs.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffers {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )


    def add(
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
        to_secondary: bool = False,
    ) -> None:
        """
        Add a new sample to the replay buffer.

        :param obs: Observation of the current state.
        :param next_obs: Observation of the next state.
        :param action: Action taken.
        :param reward: Reward received.
        :param done: Whether the episode has ended.
        :param infos: Additional information.
        :param to_secondary: Whether to add the sample to the secondary buffer.
        """
        if to_secondary:
            self.secondary_buffer.add(obs, next_obs, action, reward, done, infos)
        else:
            self.primary_buffer.add(obs, next_obs, action, reward, done, infos)


    def sample(
        self, 
        batch_size: int, 
        env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffers.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: A batch of combined samples from the replay buffers.
        """
        primary_size = 0
        secondary_size = 0
        primary_samples = None
        secondary_samples = None

        if self.primary_buffer is not None:
            primary_size = (
                self.primary_buffer.buffer_size 
                if self.primary_buffer.full 
                else self.primary_buffer.pos
            )
        if self.secondary_buffer is not None:
            secondary_size = (
                self.secondary_buffer.buffer_size 
                if self.secondary_buffer.full 
                else self.secondary_buffer.pos
            )

        total_size = primary_size + secondary_size

        batch_inds = np.random.randint(0, total_size, size=batch_size)

        primary_batch_inds = batch_inds[batch_inds < primary_size]
        secondary_batch_inds = batch_inds[batch_inds >= primary_size] - primary_size

        if len(primary_batch_inds) > 0:
            primary_samples = self.primary_buffer._get_samples(
                primary_batch_inds, env=env
            )
        
        if len(secondary_batch_inds) > 0:
            secondary_samples = self.secondary_buffer._get_samples(
                secondary_batch_inds, env=env
            )

        return DoubleReplayBuffer._concat_samples(
            primary_samples, secondary_samples
        )

    @staticmethod
    def _concat_samples(
        samples_1: ReplayBufferSamples,
        samples_2: ReplayBufferSamples,
    ) -> ReplayBufferSamples:
        """
        Concatenate two sets of samples.

        :param samples_1: First set of samples.
        :param samples_2: Second set of samples.
        :return: Concatenated samples.
        """
        if samples_1 is None:
            return samples_2
        elif samples_2 is None:
            return samples_1
        
        def _concat_tensor(
                tensor_1: th.Tensor | dict, 
                tensor_2: th.Tensor | dict
            ) -> th.Tensor | dict:
            if isinstance(tensor_1, dict) and isinstance(tensor_2, dict):
                if set(tensor_1.keys()) != set(tensor_2.keys()):
                    raise ValueError(
                        f"Keys do not match: {tensor_1.keys()} != {tensor_2.keys()}"
                    )

                return {
                    key: th.cat([
                        tensor_1[key], tensor_2[key]
                    ]) 
                    for key in tensor_1.keys()
                }
            return th.cat([tensor_1, tensor_2])
        
        if type(samples_1) != type(samples_2):
            raise ValueError(
                f"Samples do not match: {type(samples_1)} != {type(samples_2)}"
            )
        
        keys_1 = samples_1._fields
        keys_2 = samples_2._fields
        if keys_1 != keys_2:
            raise ValueError(
                f"Keys do not match: {keys_1} != {keys_2}"
            )
        
        return samples_1.__class__(
            **{
                key: _concat_tensor(
                    getattr(samples_1, key), 
                    getattr(samples_2, key)
                )
                for key in keys_1
            }
        )

class DoubleReplayBufferDQN(DoubleDQN):
    """
    A DQN agent that uses a double replay buffer.
    
    Parameters:
        total_buffer_size (int): The maximum size of the replay buffer.
        secondary_buffer_fraction (float): The size of the secondary buffer as 
            a fraction of the total buffer size.
        observation_space (gym.Space): The observation space of the environment.
        action_space (gym.Space): The action space of the environment.
        device (str): The device to store the buffer on (e.g., 'cpu' or 'cuda').
        n_envs (int): The number of environments.
        optimize_memory_usage (bool): Whether to optimize memory usage.
        handle_timeout_termination (bool): Whether to handle timeout termination.
    """
    
    def __init__(
        self,
        total_buffer_size: int,
        secondary_buffer_fraction: float,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.replay_buffer = DoubleReplayBuffer(
            total_buffer_size=total_buffer_size,
            secondary_buffer_fraction=secondary_buffer_fraction,
            replay_buffer_class=DictReplayBuffer,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            n_envs=self.n_envs
        )
        self.replay_buffer_class = DoubleReplayBuffer

    def _truncate_replay_buffer(self, buffer: DictReplayBuffer, size: int):
        """
        Truncate the replay buffer to a specific size. Takes the first `size` samples
        and discards the rest. Does not modify the buffer if it is already smaller than `size`.

        :param buffer: The replay buffer to truncate.
        :param size: The size to truncate to.
        """
        if buffer.buffer_size > size:
            # truncate the buffer to the new size
            buffer.buffer_size = size
            if buffer.pos >= size:
                buffer.full = True
                buffer.pos = 0
            
            for k in buffer.observations:
                buffer.observations[k] = buffer.observations[k][:size].copy()
                buffer.next_observations[k] = buffer.next_observations[k][:size].copy()

            buffer.actions = buffer.actions[:size].copy()
            buffer.rewards = buffer.rewards[:size].copy()
            buffer.dones = buffer.dones[:size].copy()
            buffer.timeouts = buffer.timeouts[:size].copy()
    
    def load_replay_buffer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        buffer: str = "secondary",
    ) -> None:
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        """
        if buffer == "primary":
            old_size = self.replay_buffer.primary_buffer.buffer_size
        else:
            old_size = self.replay_buffer.secondary_buffer.buffer_size

        replay_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(replay_buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"

        self._truncate_replay_buffer(
            replay_buffer, old_size
        )

        # Backward compatibility with SB3 < 2.1.0 replay buffer
        # Keep old behavior: do not handle timeout termination separately
        if not hasattr(replay_buffer, "handle_timeout_termination"):  # pragma: no cover
            replay_buffer.handle_timeout_termination = False
            replay_buffer.timeouts = np.zeros_like(replay_buffer.dones)
        
        # Update saved replay buffer device to match current setting, see GH#1561
        replay_buffer.device = self.device
        
        if buffer == "primary":
            self.replay_buffer.primary_buffer = replay_buffer
        else:
            self.replay_buffer.secondary_buffer = replay_buffer

if __name__ == "__main__":
    # Example usage
    pass
