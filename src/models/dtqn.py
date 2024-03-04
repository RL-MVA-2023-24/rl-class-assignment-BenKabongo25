from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from enum import Enum
from time import time, sleep
from tqdm import tqdm
from typing import Callable, Dict, Optional, Tuple, Union

import argparse
import csv
import gymnasium as gym
import joblib
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.MultiheadAttention):
        module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
        module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.in_proj_bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class RNG:
    rng: np.random.Generator = None


class EpsilonAnneal(ABC):
    @abstractmethod
    def anneal(self):
        pass


class Constant(EpsilonAnneal):
    def __init__(self, start):
        self.val = start

    def anneal(self):
        pass


class LinearAnneal(EpsilonAnneal):

    def __init__(self, start: float, end: float, duration: int):
        self.val = start
        self.min = end
        self.duration = duration

    def anneal(self):
        self.val = max(self.min, self.val - (self.val - self.min) / self.duration)


def set_global_seed(seed: int, *args: Tuple[gym.Env]) -> None:
    random.seed(seed)
    tseed = random.randint(1, 1e6)
    npseed = random.randint(1, 1e6)
    ospyseed = random.randint(1, 1e6)
    torch.manual_seed(tseed)
    np.random.seed(npseed)
    for env in args:
        #env.seed(seed=seed)
        env.observation_space.seed(seed=seed)
        env.action_space.seed(seed=seed)
    os.environ["PYTHONHASHSEED"] = str(ospyseed)
    RNG.rng = np.random.Generator(np.random.PCG64(seed=seed))


class Bag:

    def __init__(self, bag_size: int, obs_mask: Union[int, float], obs_length: int):
        self.size = bag_size
        self.obs_mask = obs_mask
        self.obs_length = obs_length
        self.pos = 0
        self.obss, self.actions = self.make_empty_bag()

    def reset(self) -> None:
        self.pos = 0
        self.obss, self.actions = self.make_empty_bag()

    def add(self, obs: np.ndarray, action: int) -> bool:
        if not self.is_full:
            self.obss[self.pos] = obs
            self.actions[self.pos] = action
            self.pos += 1
            return True
        else:
            return False

    def export(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.obss[: self.pos], self.actions[: self.pos]

    def make_empty_bag(self) -> np.ndarray:
        return np.full((self.size, self.obs_length), self.obs_mask), np.full((self.size, 1), 0)

    @property
    def is_full(self) -> bool:
        return self.pos >= self.size


class Context:

    def __init__(
        self,
        context_length: int,
        obs_mask: int,
        num_actions: int,
        env_obs_length: int,
        init_hidden: Tuple[torch.Tensor] = None,
    ):
        self.max_length = context_length
        self.env_obs_length = env_obs_length
        self.num_actions = num_actions
        self.obs_mask = obs_mask
        self.reward_mask = 0.0
        self.done_mask = True
        self.timestep = 0
        self.init_hidden = init_hidden

    def reset(self, obs: np.ndarray):
        self.obs = np.full([self.max_length, self.env_obs_length], self.obs_mask)
        self.obs[0] = obs
        self.action = RNG.rng.integers(self.num_actions, size=(self.max_length, 1))
        self.reward = np.full_like(self.action, self.reward_mask)
        self.done = np.full_like(self.reward, self.done_mask, dtype=np.int32)
        self.hidden = self.init_hidden
        self.timestep = 0

    def add_transition(
        self, o: np.ndarray, a: int, r: float, done: bool
    ) -> Tuple[Union[np.ndarray, None], Union[int, None]]:
        self.timestep += 1
        self.obs = self.roll(self.obs)
        self.action = self.roll(self.action)
        self.reward = self.roll(self.reward)
        self.done = self.roll(self.done)

        t = min(self.timestep, self.max_length - 1)

        evicted_obs = None
        evicted_action = None
        if self.is_full:
            evicted_obs = self.obs[t].copy()
            evicted_action = self.action[t]

        self.obs[t] = o
        self.action[t] = np.array([a])
        self.reward[t] = np.array([r])
        self.done[t] = np.array([done])

        return evicted_obs, evicted_action

    def export(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        current_timestep = min(self.timestep, self.max_length) - 1
        return (
            self.obs[current_timestep + 1],
            self.action[current_timestep],
            self.reward[current_timestep],
            self.done[current_timestep],
        )

    def roll(self, arr: np.ndarray) -> np.ndarray:
        return np.roll(arr, -1, axis=0) if self.timestep >= self.max_length else arr

    @property
    def is_full(self) -> bool:
        return self.timestep >= self.max_length

    @staticmethod
    def context_like(context):
        return Context(
            context.max_length,
            context.obs_mask,
            context.num_actions,
            context.env_obs_length,
            init_hidden=context.init_hidden,
        )


class ReplayBuffer:

    def __init__(
        self,
        buffer_size: int,
        env_obs_length: Union[int, Tuple],
        obs_mask: int,
        max_episode_steps: int,
        context_len: Optional[int]=1,
    ):
        self.max_size = buffer_size // max_episode_steps
        self.context_len = context_len
        self.env_obs_length = env_obs_length
        self.max_episode_steps = max_episode_steps
        self.obs_mask = obs_mask
        self.pos = [0, 0]

        self.obss = np.full(
            [self.max_size, max_episode_steps + 1, env_obs_length],
            obs_mask,
            dtype=np.float32,
        )

        # Need the +1 so we have space to roll for the first observation
        self.actions = np.zeros(
            [self.max_size, max_episode_steps + 1, 1],
            dtype=np.uint8,
        )
        self.rewards = np.zeros(
            [self.max_size, max_episode_steps, 1],
            dtype=np.float32,
        )
        self.dones = np.ones(
            [self.max_size, max_episode_steps, 1],
            dtype=np.bool_,
        )
        self.episode_lengths = np.zeros([self.max_size], dtype=np.uint8)

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        episode_length: Optional[int] = 0,
    ) -> None:
        episode_idx = self.pos[0] % self.max_size
        obs_idx = self.pos[1]
        self.obss[episode_idx, obs_idx + 1] = obs
        self.actions[episode_idx, obs_idx] = action
        self.rewards[episode_idx, obs_idx] = reward
        self.dones[episode_idx, obs_idx] = done
        self.episode_lengths[episode_idx] = episode_length
        self.pos = [self.pos[0], self.pos[1] + 1]

    def store_obs(self, obs: np.ndarray) -> None:
        episode_idx = self.pos[0] % self.max_size
        self.cleanse_episode(episode_idx)
        self.obss[episode_idx, 0] = obs

    def can_sample(self, batch_size: int) -> bool:
        return batch_size < self.pos[0]

    def flush(self):
        self.pos = [self.pos[0] + 1, 0]

    def cleanse_episode(self, episode_idx: int) -> None:
        self.obss[episode_idx] = np.full(
            [self.max_episode_steps + 1, self.env_obs_length],
            self.obs_mask,
            dtype=np.float32,
        )
        self.actions[episode_idx] = np.zeros(
            [self.max_episode_steps + 1, 1],
            dtype=np.uint8,
        )
        self.rewards[episode_idx] = np.zeros(
            [self.max_episode_steps, 1],
            dtype=np.float32,
        )
        self.dones[episode_idx] = np.ones(
            [self.max_episode_steps, 1],
            dtype=np.bool_,
        )
        self.episode_lengths[episode_idx] = 0

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        valid_episodes = [
            i
            for i in range(min(self.pos[0], self.max_size))
            if i != self.pos[0] % self.max_size
        ]
        episode_idxes = np.array(
            [[random.choice(valid_episodes)] for _ in range(batch_size)]
        )
        transition_starts = np.array(
            [
                random.randint(
                    0, max(0, self.episode_lengths[idx[0]] - self.context_len)
                )
                for idx in episode_idxes
            ]
        )
        transitions = np.array(
            [range(start, start + self.context_len) for start in transition_starts]
        )
        return (
            self.obss[episode_idxes, transitions],
            self.actions[episode_idxes, transitions],
            self.rewards[episode_idxes, transitions],
            self.obss[episode_idxes, 1 + transitions],
            self.actions[episode_idxes, 1 + transitions],
            self.dones[episode_idxes, transitions],
            np.clip(self.episode_lengths[episode_idxes], 0, self.context_len),
        )

    def sample_with_bag(
        self, batch_size: int, sample_bag: Bag
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        episode_idxes = np.array(
            [
                [
                    random.choice(
                        [
                            i
                            for i in range(min(self.pos[0], self.max_size))
                            if i != self.pos[0]
                        ]
                    )
                ]
                for _ in range(batch_size)
            ]
        )
        transition_starts = np.array(
            [
                random.randint(
                    0, max(0, self.episode_lengths[idx[0]] - self.context_len)
                )
                for idx in episode_idxes
            ]
        )
        transitions = np.array(
            [range(start, start + self.context_len) for start in transition_starts]
        )

        bag_obss = np.full(
            [batch_size, sample_bag.size, sample_bag.obs_length],
            sample_bag.obs_mask,
        )
        bag_actions = np.full(
            [batch_size, sample_bag.size, 1],
            0,
        )

        for bag_idx in range(batch_size):
            if transition_starts[bag_idx] < sample_bag.size:
                bag_obss[bag_idx, : transition_starts[bag_idx]] = self.obss[
                    episode_idxes[bag_idx], : transition_starts[bag_idx]
                ]
                bag_actions[bag_idx, : transition_starts[bag_idx]] = self.actions[
                    episode_idxes[bag_idx], : transition_starts[bag_idx]
                ]
            else:
                bag_obss[bag_idx] = np.array(
                    random.sample(
                        self.obss[episode_idxes[bag_idx], : transition_starts[bag_idx]]
                        .squeeze()
                        .tolist(),
                        k=sample_bag.size,
                    )
                )
                bag_actions[bag_idx] = np.expand_dims(
                    np.array(
                        random.sample(
                            self.actions[
                                episode_idxes[bag_idx], : transition_starts[bag_idx]
                            ]
                            .squeeze()
                            .tolist(),
                            k=sample_bag.size,
                        )
                    ),
                    axis=1,
                )
        return (
            self.obss[episode_idxes, transitions],
            self.actions[episode_idxes, transitions],
            self.rewards[episode_idxes, transitions],
            self.obss[episode_idxes, 1 + transitions],
            self.actions[episode_idxes, 1 + transitions],
            self.dones[episode_idxes, transitions],
            self.episode_lengths[episode_idxes],
            bag_obss,
            bag_actions,
        )


class RunningAverage:

    def __init__(self, size):
        self.size = size
        self.q = deque()
        self.sum = 0

    def add(self, val):
        self.q.append(val)
        self.sum += val
        if len(self.q) > self.size:
            self.sum -= self.q.popleft()

    def mean(self):
        return self.sum / max(len(self.q), 1)


def timestamp():
    return datetime.now().strftime("%B %d, %H:%M:%S")


class CSVLogger:

    def __init__(self, path: str, args: argparse.Namespace, envs: Tuple[gym.Env]):
        self.results_path = path + "_results.csv"
        self.losses_path = path + "_losses.csv"
        self.envs = envs
        if not os.path.exists(self.results_path):
            head_row = ["Hours", "Step"]
            for i, env in enumerate(self.envs):
                head_row += [
                    f"{i}/SuccessRate",
                    f"{i}/EpisodeLength",
                    f"{i}/Return",
                ]
            with open(self.results_path, "w") as file:
                writer = csv.writer(file)
                writer.writerow(head_row)
        if not os.path.exists(self.losses_path):
            with open(self.losses_path, "w") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "Hours",
                        "Step",
                        "TD Error",
                        "Grad Norm",
                        "Max Q Value",
                        "Mean Q Value",
                        "Min Q Value",
                        "Max Target Value",
                        "Mean Target Value",
                        "Min Target Value",
                    ]
                )

    def log(self, results: Dict[str, str], step: int):
        results_row = [results["losses/hours"], step]
        for i, env in enumerate(self.envs):
            results_row += [
                results[f"{i}/SuccessRate"],
                results[f"{i}/EpisodeLength"],
                results[f"{i}/Return"],
            ]
        with open(self.results_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow(results_row)
        with open(self.losses_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    results["losses/hours"],
                    step,
                    results["losses/TD_Error"],
                    results["losses/Grad_Norm"],
                    results["losses/Max_Q_Value"],
                    results["losses/Mean_Q_Value"],
                    results["losses/Min_Q_Value"],
                    results["losses/Max_Target_Value"],
                    results["losses/Mean_Target_Value"],
                    results["losses/Min_Target_Value"],
                ]
            )


def get_logger(policy_path: str, args: argparse.Namespace, envs: Tuple[gym.Env]):
    logger = CSVLogger(policy_path, args, envs)
    return logger


class ObservationEmbeddingRepresentation(nn.Module):

    def __init__(self, observation_embedding: nn.Module):
        super().__init__()
        self.observation_embedding = observation_embedding

    def forward(self, obs: torch.Tensor):
        batch, seq = obs.size(0), obs.size(1)
        obs = torch.flatten(obs, start_dim=0, end_dim=1)
        obs_embed = self.observation_embedding(obs)
        obs_embed = obs_embed.reshape(batch, seq, obs_embed.size(-1))
        return obs_embed

    @staticmethod
    def make_action_representation(num_actions: int, action_dim: int) -> ObservationEmbeddingRepresentation:
        embed = nn.Sequential(
            nn.Embedding(num_actions, action_dim), nn.Flatten(start_dim=-2)
        )
        return ObservationEmbeddingRepresentation(observation_embedding=embed)

    @staticmethod
    def make_continuous_representation(obs_dim: int, outer_embed_size: int):
        embedding = nn.Linear(obs_dim, outer_embed_size)
        return ObservationEmbeddingRepresentation(observation_embedding=embedding)


class ActionEmbeddingRepresentation(nn.Module):
    def __init__(self, num_actions: int, action_dim: int):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(num_actions, action_dim),
            nn.Flatten(start_dim=-2),
        )

    def forward(self, action: torch.Tensor):
        return self.embedding(action)


class PosEnum(Enum):
    LEARNED = "learned"
    SIN = "sin"
    NONE = "none"


class PositionEncoding(nn.Module):
    def __init__(self, position_encoding: nn.Module):
        super().__init__()
        self.position_encoding = position_encoding

    def forward(self):
        return self.position_encoding

    @staticmethod
    def make_sinusoidal_position_encoding(context_len: int, embed_dim: int) -> PositionEncoding:
        position = torch.arange(context_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim)
        )
        pos_encoding = torch.zeros(1, context_len, embed_dim)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        return PositionEncoding(nn.Parameter(pos_encoding, requires_grad=False))

    @staticmethod
    def make_learned_position_encoding(context_len: int, embed_dim: int) -> PositionEncoding:
        return PositionEncoding(
            nn.Parameter(torch.zeros(1, context_len, embed_dim), requires_grad=True)
        )

    @staticmethod
    def make_empty_position_encoding(context_len: int, embed_dim: int) -> PositionEncoding:
        return PositionEncoding(
            nn.Parameter(torch.zeros(1, context_len, embed_dim), requires_grad=False)
        )


class GRUGate(nn.Module):

    def __init__(self, embed_size: int):
        super().__init__()
        self.w_r = nn.Linear(embed_size, embed_size, bias=False)
        self.u_r = nn.Linear(embed_size, embed_size, bias=False)
        self.w_z = nn.Linear(embed_size, embed_size)
        self.u_z = nn.Linear(embed_size, embed_size, bias=False)
        self.w_g = nn.Linear(embed_size, embed_size, bias=False)
        self.u_g = nn.Linear(embed_size, embed_size, bias=False)
        self.init_bias()

    def init_bias(self):
        with torch.no_grad():
            self.w_z.bias.fill_(-2)

    def forward(self, x, y):
        z = torch.sigmoid(self.w_z(y) + self.u_z(x))
        r = torch.sigmoid(self.w_r(y) + self.u_r(x))
        h = torch.tanh(self.w_g(y) + self.u_g(r * x))
        return (1.0 - z) * x + z * h


class ResGate(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y


class TransformerLayer(nn.Module):

    def __init__(
        self,
        num_heads: int,
        embed_size: int,
        history_len: int,
        dropout: float,
        attn_gate,
        mlp_gate
    ):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.layernorm2 = nn.LayerNorm(embed_size)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )
        self.attn_gate = attn_gate
        self.mlp_gate = mlp_gate
        self.alpha = None

        self.attn_mask = nn.Parameter(
            torch.triu(torch.ones(history_len, history_len), diagonal=1),
            requires_grad=False,
        )
        self.attn_mask[self.attn_mask.bool()] = -float("inf")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention, self.alpha = self.attention(
            x,
            x,
            x,
            attn_mask=self.attn_mask[: x.size(1), : x.size(1)],
            average_attn_weights=True
        )
        x = self.attn_gate(x, F.relu(attention))
        x = self.layernorm1(x)
        ffn = self.ffn(x)
        x = self.mlp_gate(x, F.relu(ffn))
        x = self.layernorm2(x)
        return x


class TransformerIdentityLayer(TransformerLayer):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm1 = self.layernorm1(x)
        attention, self.alpha = self.attention(
            x_norm1,
            x_norm1,
            x_norm1,
            attn_mask=self.attn_mask[: x_norm1.size(1), : x_norm1.size(1)],
            average_attn_weights=True
        )
        x = self.attn_gate(x, F.relu(attention))
        x_norm2 = self.layernorm2(x)
        ffn = self.ffn(x_norm2)
        x = self.mlp_gate(x, F.relu(ffn))
        return x


class DTQN(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        action_dim: int,
        inner_embed_size: int,
        num_heads: int,
        num_layers: int,
        history_len: int,
        dropout: float=0.0,
        gate: str="res",
        identity: bool=False,
        pos: Union[str, int]=1,
        bag_size: int=0
    ):
        super().__init__()
        self.obs_dim = obs_dim
        obs_output_dim = inner_embed_size - action_dim
        if action_dim > 0:
            self.action_embedding = ActionEmbeddingRepresentation(
                num_actions=num_actions, action_dim=action_dim
            )
        else:
            self.action_embedding = None

        self.obs_embedding = (
            ObservationEmbeddingRepresentation.make_continuous_representation(
                obs_dim=obs_dim, outer_embed_size=obs_output_dim
            )
        )

        pos_function_map = {
            PosEnum.LEARNED: PositionEncoding.make_learned_position_encoding,
            PosEnum.SIN: PositionEncoding.make_sinusoidal_position_encoding,
            PosEnum.NONE: PositionEncoding.make_empty_position_encoding,
        }
        self.position_embedding = pos_function_map[PosEnum(pos)](
            context_len=history_len, embed_dim=inner_embed_size
        )

        self.dropout = nn.Dropout(dropout)

        if gate == "gru":
            attn_gate = GRUGate(embed_size=inner_embed_size)
            mlp_gate = GRUGate(embed_size=inner_embed_size)
        elif gate == "res":
            attn_gate = ResGate()
            mlp_gate = ResGate()
        else:
            raise ValueError("Gate must be one of `gru`, `res`")

        if identity:
            transformer_block = TransformerIdentityLayer
        else:
            transformer_block = TransformerLayer
        self.transformer_layers = nn.Sequential(
            *[
                transformer_block(
                    num_heads,
                    inner_embed_size,
                    history_len,
                    dropout,
                    attn_gate,
                    mlp_gate,
                )
                for _ in range(num_layers)
            ]
        )

        self.bag_size = bag_size
        self.bag_attn_weights = None
        if bag_size > 0:
            self.bag_attention = nn.MultiheadAttention(
                inner_embed_size,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.ffn = nn.Sequential(
                nn.Linear(inner_embed_size * 2, inner_embed_size),
                nn.ReLU(),
                nn.Linear(inner_embed_size, num_actions),
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(inner_embed_size, inner_embed_size),
                nn.ReLU(),
                nn.Linear(inner_embed_size, num_actions),
            )

        self.history_len = history_len
        self.apply(init_weights)

    def forward(
        self,
        obss: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        bag_obss: Optional[torch.Tensor] = None,
        bag_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # obss    is  batch x seq_len  x obs_dim
        # actions is  batch x seq_len  x       1
        # bag     is  batch x bag_size x obs_dim
        
        history_len = obss.size(1)
        assert (
            history_len <= self.history_len
        ), "Cannot forward, history is longer than expected."

        token_embeddings = self.obs_embedding(obss)

        # Just to keep shapes correct if we choose to disble including actions
        if self.action_embedding is not None:
            # [batch x seq_len x 1] -> [batch x seq_len x action_embed]
            action_embed = self.action_embedding(actions)

            if history_len > 1:
                action_embed = torch.roll(action_embed, 1, 1)
                # First observation in the sequence doesn't have a previous action, so zero the features
                action_embed[:, 0, :] = 0.0
            token_embeddings = torch.concat([action_embed, token_embeddings], dim=-1)

        # [batch x seq_len x model_embed] -> [batch x seq_len x model_embed]
        working_memory = self.transformer_layers(
            self.dropout(
                token_embeddings + self.position_embedding()[:, :history_len, :]
            )
        )

        if self.bag_size > 0:
            # [batch x bag_size x action_embed] + [batch x bag_size x obs_embed] -> [batch x bag_size x model_embed]
            if self.action_embedding is not None:
                bag_embeddings = torch.concat(
                    [self.action_embedding(bag_actions), self.obs_embedding(bag_obss)],
                    dim=-1,
                )
            else:
                bag_embeddings = self.obs_embedding(bag_obss)
            # [batch x seq_len x model_embed] x [batch x bag_size x model_embed] -> [batch x seq_len x model_embed]
            persistent_memory, self.attn_weights = self.bag_attention(
                working_memory, bag_embeddings, bag_embeddings
            )
            output = self.ffn(torch.concat([working_memory, persistent_memory], dim=-1))
        else:
            output = self.ffn(working_memory)

        return output[:, -history_len:, :]


class TrainMode(Enum):
    TRAIN = 1
    EVAL = 2


class DqnAgent:
    def __init__(
        self,
        network_factory: Callable[[], nn.Module],
        buffer_size: int,
        device: torch.device,
        env_obs_length: int,
        max_env_steps: int,
        obs_mask: Union[int, float],
        num_actions: int,
        learning_rate: float=0.0003,
        batch_size: int=32,
        context_len: int=1,
        gamma: float=0.99,
        grad_norm_clip: float=1.0,
        target_update_frequency: int=10_000,
        **kwargs,
    ):
        self.context_len = context_len
        self.env_obs_length = env_obs_length
        
        self.policy_network = network_factory()
        self.target_network = network_factory()
        
        self.target_update()
        self.target_network.eval()

        self.obs_context_type = np.float32
        self.obs_tensor_type = torch.float32

        self.device = device

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(
            buffer_size,
            env_obs_length=env_obs_length,
            obs_mask=obs_mask,
            max_episode_steps=max_env_steps,
            context_len=context_len,
        )

        self.batch_size = batch_size
        self.gamma = gamma
        self.grad_norm_clip = grad_norm_clip
        self.target_update_frequency = target_update_frequency

        # Logging
        self.num_train_steps = 0
        self.td_errors = RunningAverage(100)
        self.grad_norms = RunningAverage(100)
        self.qvalue_max = RunningAverage(100)
        self.target_max = RunningAverage(100)
        self.qvalue_mean = RunningAverage(100)
        self.target_mean = RunningAverage(100)
        self.qvalue_min = RunningAverage(100)
        self.target_min = RunningAverage(100)

        self.num_actions = num_actions
        self.train_mode = TrainMode.TRAIN
        self.obs_mask = obs_mask

        self.train_context = Context(
            context_len, obs_mask, self.num_actions, env_obs_length
        )
        self.eval_context = Context(
            context_len, obs_mask, self.num_actions, env_obs_length
        )

    @property
    def context(self) -> Context:
        if self.train_mode == TrainMode.TRAIN:
            return self.train_context
        elif self.train_mode == TrainMode.EVAL:
            return self.eval_context

    def eval_on(self) -> None:
        self.train_mode = TrainMode.EVAL
        self.policy_network.eval()

    def eval_off(self) -> None:
        self.train_mode = TrainMode.TRAIN
        self.policy_network.train()

    @torch.no_grad()
    def get_action(self, epsilon=0.0) -> int:
        """Use policy_network to get an e-greedy action given the current obs."""
        if RNG.rng.random() < epsilon:
            return RNG.rng.integers(self.num_actions)
        q_values = self.policy_network(
            torch.as_tensor(
                self.context.obs[min(self.context.timestep, self.context_len - 1)],
                dtype=self.obs_tensor_type,
                device=self.device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return torch.argmax(q_values).item()

    def observe(self, obs, action, reward, done) -> None:
        if self.train_mode == TrainMode.TRAIN:
            self.replay_buffer.store(obs, action, reward, done)

    def context_reset(self, obs: np.ndarray) -> None:
        self.context.reset(obs)
        if self.train_mode == TrainMode.TRAIN:
            self.replay_buffer.store_obs(obs)

    def train(self) -> None:
        """Perform one gradient step of the network"""
        if not self.replay_buffer.can_sample(self.batch_size):
            return

        self.eval_off()
        obss, actions, rewards, next_obss, _, dones, _ = self.replay_buffer.sample(
            self.batch_size
        )

        # We pull obss/next_obss as [batch-size x 1 x obs-dim]
        obss = torch.as_tensor(obss, dtype=self.obs_tensor_type, device=self.device)
        next_obss = torch.as_tensor(
            next_obss, dtype=self.obs_tensor_type, device=self.device
        )
        # Actions is [batch-size x 1 x 1] which we want to be [batch-size x 1]
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        # Rewards/Dones are [batch-size x 1 x 1] which we want to be [batch-size]
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        ).squeeze()
        dones = torch.as_tensor(dones, dtype=torch.long, device=self.device).squeeze()

        # obss is [batch-size x obs-dim] and after network is [batch-size x action-dim]
        # Then we gather it and squeeze to [batch-size]
        q_values = self.policy_network(obss)
        # [batch-seq-actions]
        q_values = q_values.gather(2, actions).squeeze()

        with torch.no_grad():
            # We use DDQN, so the policy network determines which future actions we'd
            # take, but the target network determines the value of those
            next_obs_qs = self.policy_network(next_obss)
            argmax = torch.argmax(next_obs_qs, axis=-1).unsqueeze(-1)
            next_obs_q_values = (
                self.target_network(next_obss).gather(2, argmax).squeeze()
            )

            # here goes BELLMAN
            targets = rewards + (1 - dones) * (next_obs_q_values * self.gamma)

        self.qvalue_max.add(q_values.max().item())
        self.qvalue_mean.add(q_values.mean().item())
        self.qvalue_min.add(q_values.min().item())

        self.target_max.add(targets.max().item())
        self.target_mean.add(targets.mean().item())
        self.target_min.add(targets.min().item())

        # Optimization step
        loss = F.mse_loss(q_values, targets)
        self.td_errors.add(loss.item())
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            self.grad_norm_clip,
            error_if_nonfinite=True,
        )
        self.grad_norms.add(norm.item())
        self.optimizer.step()
        self.num_train_steps += 1

        if self.num_train_steps % self.target_update_frequency == 0:
            self.target_update()

    def target_update(self) -> None:
        """Hard update where we copy the network parameters from the policy network to the target network"""
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def save_mini_checkpoint(self, checkpoint_dir: str) -> None:
        torch.save(
            {"step": self.num_train_steps},
            checkpoint_dir + "_mini_checkpoint.pt",
        )

    @staticmethod
    def load_mini_checkpoint(checkpoint_dir: str) -> dict:
        return torch.load(checkpoint_dir + "_mini_checkpoint.pt")

    def save_checkpoint(
        self,
        checkpoint_dir: str,
        episode_successes: RunningAverage,
        episode_rewards: RunningAverage,
        episode_lengths: RunningAverage,
        eps: LinearAnneal,
    ) -> None:
        self.save_mini_checkpoint(checkpoint_dir=checkpoint_dir)
        torch.save(
            {
                "step": self.num_train_steps,
                # Replay Buffer: Don't keep the observation index saved
                "replay_buffer_pos": [self.replay_buffer.pos[0], 0],
                # Neural Net
                "policy_net_state_dict": self.policy_network.state_dict(),
                "target_net_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": eps.val,
                # Results
                "episode_successes": episode_successes,
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                # Losses
                "td_errors": self.td_errors,
                "grad_norms": self.grad_norms,
                "qvalue_max": self.qvalue_max,
                "qvalue_mean": self.qvalue_mean,
                "qvalue_min": self.qvalue_min,
                "target_max": self.target_max,
                "target_mean": self.target_mean,
                "target_min": self.target_min,
                # RNG states
                "random_rng_state": random.getstate(),
                "rng_bit_generator_state": RNG.rng.bit_generator.state,
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state": torch.cuda.get_rng_state()
                if torch.cuda.is_available()
                else torch.get_rng_state(),
            },
            checkpoint_dir + "_checkpoint.pt",
        )
        joblib.dump(self.replay_buffer.obss, checkpoint_dir + "buffer_obss.sav")
        joblib.dump(self.replay_buffer.actions, checkpoint_dir + "buffer_actions.sav")
        joblib.dump(self.replay_buffer.rewards, checkpoint_dir + "buffer_rewards.sav")
        joblib.dump(self.replay_buffer.dones, checkpoint_dir + "buffer_dones.sav")
        joblib.dump(
            self.replay_buffer.episode_lengths, checkpoint_dir + "buffer_eplens.sav"
        )

    def load_checkpoint(
        self, checkpoint_dir: str
    ) -> Tuple[str, RunningAverage, RunningAverage, RunningAverage, float]:
        checkpoint = torch.load(checkpoint_dir + "_checkpoint.pt")
        # checkpoint = np.load(checkpoint_dir + "_checkpoint.npz", allow_pickle=True)

        self.num_train_steps = checkpoint["step"]
        # Replay Buffer
        self.replay_buffer.pos = checkpoint["replay_buffer_pos"]
        self.replay_buffer.obss = joblib.load(checkpoint_dir + "buffer_obss.sav")
        self.replay_buffer.actions = joblib.load(checkpoint_dir + "buffer_actions.sav")
        self.replay_buffer.rewards = joblib.load(checkpoint_dir + "buffer_rewards.sav")
        self.replay_buffer.dones = joblib.load(checkpoint_dir + "buffer_dones.sav")
        self.replay_buffer.episode_lengths = joblib.load(
            checkpoint_dir + "buffer_eplens.sav"
        )
        # Neural Net
        self.policy_network.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Losses
        self.td_errors = checkpoint["td_errors"]
        self.grad_norms = checkpoint["grad_norms"]
        self.qvalue_max = checkpoint["qvalue_max"]
        self.qvalue_mean = checkpoint["qvalue_mean"]
        self.qvalue_min = checkpoint["qvalue_min"]
        self.target_max = checkpoint["target_max"]
        self.target_mean = checkpoint["target_mean"]
        self.target_min = checkpoint["target_min"]
        # RNG states
        random.setstate(checkpoint["random_rng_state"])
        RNG.rng.bit_generator.state = checkpoint["rng_bit_generator_state"]
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint["torch_cuda_rng_state"])

        # Results
        episode_successes = checkpoint["episode_successes"]
        episode_rewards = checkpoint["episode_rewards"]
        episode_lengths = checkpoint["episode_lengths"]
        # Exploration value
        epsilon = checkpoint["epsilon"]

        return (
            episode_successes,
            episode_rewards,
            episode_lengths,
            epsilon,
        )


class DtqnAgent(DqnAgent):
    def __init__(
        self,
        network_factory: Callable[[], nn.Module],
        buffer_size: int,
        device: torch.device,
        env_obs_length: int,
        max_env_steps: int,
        obs_mask: Union[int, float],
        num_actions: int,
        learning_rate: float=0.0003,
        batch_size: int=32,
        context_len: int=50,
        gamma: float=0.99,
        grad_norm_clip: float=1.0,
        target_update_frequency: int=10_000,
        history: int=50,
        bag_size: int=0,
        **kwargs,
    ):
        super().__init__(
            network_factory,
            buffer_size,
            device,
            env_obs_length,
            max_env_steps,
            obs_mask,
            num_actions,
            learning_rate,
            batch_size,
            context_len,
            gamma,
            grad_norm_clip,
            target_update_frequency,
        )
        self.history = history
        self.train_context = Context(
            context_len,
            obs_mask,
            num_actions,
            env_obs_length,
        )
        self.eval_context = Context(
            context_len,
            obs_mask,
            num_actions,
            env_obs_length,
        )
        self.train_bag = Bag(bag_size, obs_mask, env_obs_length)
        self.eval_bag = Bag(bag_size, obs_mask, env_obs_length)

    @property
    def bag(self) -> Bag:
        if self.train_mode == TrainMode.TRAIN:
            return self.train_bag
        elif self.train_mode == TrainMode.EVAL:
            return self.eval_bag

    @torch.no_grad()
    def get_action(self, epsilon: float = 0.0) -> int:
        if RNG.rng.random() < epsilon:
            return RNG.rng.integers(self.num_actions)
        # Truncate the context of observations and actions to remove padding if it exists
        context_obs_tensor = torch.as_tensor(
            self.context.obs[: min(self.context.max_length, self.context.timestep + 1)],
            dtype=self.obs_tensor_type,
            device=self.device,
        ).unsqueeze(0)
        context_action_tensor = torch.as_tensor(
            self.context.action[
                : min(self.context.max_length, self.context.timestep + 1)
            ],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        # Always include the full bag, even if it has padding TODO:
        bag_obs_tensor = torch.as_tensor(
            self.bag.obss, dtype=self.obs_tensor_type, device=self.device
        ).unsqueeze(0)
        bag_action_tensor = torch.as_tensor(
            self.bag.actions, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        q_values = self.policy_network(
            context_obs_tensor, context_action_tensor, bag_obs_tensor, bag_action_tensor
        )

        # We take the argmax of the last timestep's Q values
        # In other words, select the highest q value action
        return torch.argmax(q_values[:, -1, :]).item()

    def context_reset(self, obs: np.ndarray) -> None:
        self.context.reset(obs)
        if self.train_mode == TrainMode.TRAIN:
            self.replay_buffer.store_obs(obs)
        if self.bag.size > 0:
            self.bag.reset()

    def observe(self, obs: np.ndarray, action: int, reward: float, done: bool) -> None:
        """Add an observation to the context. If the context would evict an observation to make room,
        attempt to put the observation in the bag, which may require evicting something else from the bag.

        If we're in train mode, then we also add the transition to our replay buffer."""
        evicted_obs, evicted_action = self.context.add_transition(
            obs, action, reward, done
        )
        # If there is an evicted obs, we need to decide if it should go in the bag or not
        if self.bag.size > 0 and evicted_obs is not None:
            # Bag is already full
            if not self.bag.add(evicted_obs, evicted_action):
                # For each possible bag, get the Q-values
                possible_bag_obss = np.tile(self.bag.obss, (self.bag.size + 1, 1, 1))
                possible_bag_actions = np.tile(
                    self.bag.actions, (self.bag.size + 1, 1, 1)
                )
                for i in range(self.bag.size):
                    possible_bag_obss[i, i] = evicted_obs
                    possible_bag_actions[i, i] = evicted_action
                tiled_context = np.tile(self.context.obs, (self.bag.size + 1, 1, 1))
                tiled_actions = np.tile(self.context.action, (self.bag.size + 1, 1, 1))
                q_values = self.policy_network(
                    torch.as_tensor(
                        tiled_context, dtype=self.obs_tensor_type, device=self.device
                    ),
                    torch.as_tensor(
                        tiled_actions, dtype=torch.long, device=self.device
                    ),
                    torch.as_tensor(
                        possible_bag_obss,
                        dtype=self.obs_tensor_type,
                        device=self.device,
                    ),
                    torch.as_tensor(
                        possible_bag_actions, dtype=torch.long, device=self.device
                    ),
                )

                bag_idx = torch.argmax(torch.mean(torch.max(q_values, 2)[0], 1))
                self.bag.obss = possible_bag_obss[bag_idx]
                self.bag.actions = possible_bag_actions[bag_idx]

        if self.train_mode == TrainMode.TRAIN:
            self.replay_buffer.store(obs, action, reward, done, self.context.timestep)

    def train(self) -> None:
        if not self.replay_buffer.can_sample(self.batch_size):
            return
        self.eval_off()
        if self.bag.size > 0:
            (
                obss,
                actions,
                rewards,
                next_obss,
                next_actions,
                dones,
                episode_lengths,
                bag_obss,
                bag_actions,
            ) = self.replay_buffer.sample_with_bag(self.batch_size, self.bag)
            # Bags: [batch-size x bag-size x obs-dim]
            bag_obss = torch.as_tensor(
                bag_obss, dtype=self.obs_tensor_type, device=self.device
            )
            bag_actions = torch.as_tensor(
                bag_actions, dtype=torch.long, device=self.device
            )
        else:
            (
                obss,
                actions,
                rewards,
                next_obss,
                next_actions,
                dones,
                episode_lengths,
            ) = self.replay_buffer.sample(self.batch_size)
            bag_obss = None
            bag_actions = None

        # Obss and Next obss: [batch-size x hist-len x obs-dim]
        obss = torch.as_tensor(obss, dtype=self.obs_tensor_type, device=self.device)
        next_obss = torch.as_tensor(
            next_obss, dtype=self.obs_tensor_type, device=self.device
        )
        # Actions: [batch-size x hist-len x 1]
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        next_actions = torch.as_tensor(
            next_actions, dtype=torch.long, device=self.device
        )
        # Rewards: [batch-size x hist-len x 1]
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        # Dones: [batch-size x hist-len x 1]
        dones = torch.as_tensor(dones, dtype=torch.long, device=self.device)

        # obss is [batch-size x hist-len x obs-len]
        # then q_values is [batch-size x hist-len x n-actions]
        q_values = self.policy_network(obss, actions, bag_obss, bag_actions)

        # After gathering, Q values becomes [batch-size x hist-len x 1] then
        # after squeeze becomes [batch-size x hist-len]
        q_values = q_values.gather(2, actions).squeeze()

        with torch.no_grad():
            # Next obss goes from [batch-size x hist-len x obs-dim] to
            # [batch-size x hist-len x n-actions] and then goes through gather and squeeze
            # to become [batch-size x hist-len]
            if self.history:
                argmax = torch.argmax(
                    self.policy_network(next_obss, next_actions, bag_obss, bag_actions),
                    dim=2,
                ).unsqueeze(-1)
                next_obs_q_values = self.target_network(
                    next_obss, next_actions, bag_obss, bag_actions
                )
                next_obs_q_values = next_obs_q_values.gather(2, argmax).squeeze()

            # here goes BELLMAN
            targets = rewards.squeeze() + (1 - dones.squeeze()) * (
                next_obs_q_values * self.gamma
            )

        q_values = q_values[:, -self.history :]
        targets = targets[:, -self.history :]
        # Calculate loss
        loss = F.mse_loss(q_values, targets)
        # Log Losses
        self.qvalue_max.add(q_values.max().item())
        self.qvalue_mean.add(q_values.mean().item())
        self.qvalue_min.add(q_values.min().item())

        self.target_max.add(targets.max().item())
        self.target_mean.add(targets.mean().item())
        self.target_min.add(targets.min().item())

        self.td_errors.add(loss.item())
        # Optimization step
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            self.grad_norm_clip,
            error_if_nonfinite=True,
        )
        # Logging
        self.grad_norms.add(norm.item())

        self.optimizer.step()
        self.num_train_steps += 1

        if self.num_train_steps % self.target_update_frequency == 0:
            self.target_update()


class ObsType(Enum):
    CONTINUOUS = 1


def get_env_obs_type(env: gym.Env) -> int:
    return ObsType.CONTINUOUS


def get_env_obs_length(env: gym.Env) -> int:
    return env.observation_space.shape[0]


def get_env_obs_mask(env: gym.Env) -> Union[int, np.ndarray]:
    return -5


def get_env_max_steps(env: gym.Env) -> Union[int, None]:
    try:
        return env._max_episode_steps
    except AttributeError:
        try:
            return env.max_episode_steps
        except AttributeError:
            return None


def get_agent(
    envs: Tuple[gym.Env],
    embed_per_obs_dim: int,
    action_dim: int,
    inner_embed: int,
    buffer_size: int,
    device: torch.device,
    learning_rate: float,
    batch_size: int,
    context_len: int,
    max_env_steps: int,
    history: int,
    target_update_frequency: int,
    gamma: float,
    num_heads: int=1,
    num_layers: int=1,
    dropout: float=0.0,
    identity: bool=False,
    gate: str="res",
    pos: str="learned",
    bag_size: int = 0,
):
    env_obs_length = get_env_obs_length(envs[0])
    env_obs_mask = get_env_obs_mask(envs[0])
    if max_env_steps <= 0:
        max_env_steps = max([get_env_max_steps(env) for env in envs])

    if history < 1 or history > context_len:
        print(
            f"History must be 1 < history <= context_len, but history is {history} and context len is {context_len}. Clipping history to {np.clip(history, 1, context_len)}..."
        )
        history = np.clip(history, 1, context_len)
    # All envs must share same action space
    num_actions = envs[0].action_space.n

    def make_dtqn():
        return lambda: DTQN(
            env_obs_length,
            num_actions,
            action_dim,
            inner_embed,
            num_heads,
            num_layers,
            context_len,
            dropout=dropout,
            gate=gate,
            identity=identity,
            pos=pos,
            bag_size=bag_size,
        ).to(device)

    network_factory = make_dtqn()

    return DtqnAgent(
        network_factory,
        buffer_size,
        device,
        env_obs_length,
        max_env_steps,
        env_obs_mask,
        num_actions,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        context_len=context_len,
        embed_size=inner_embed,
        history=history,
        target_update_frequency=target_update_frequency,
        bag_size=bag_size,
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2_000_000,
        help="Number of steps to train the agent.",
    )
    parser.add_argument(
        "--tuf",
        type=int,
        default=10_000,
        help="How many steps between each (hard) target network update.",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate for the optimizer."
    )
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--buf-size",
        type=int,
        default=500_000,
        help="Number of timesteps to store in replay buffer.",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=5_000,
        help="How many training timesteps between agent evaluations.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for each evaluation period.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Pytorch device to use."
    )
    parser.add_argument(
        "--context",
        type=int,
        default=50,
        help="The context length to use to train the network.",
    )
    parser.add_argument(
        "--obs-embed",
        type=int,
        default=8,
        help="For discrete observation domains only. The number of features to give each observation.",
    )
    parser.add_argument(
        "--a-embed",
        type=int,
        default=0,
        help="The number of features to give each action. A value of 0 will prevent the policy from using the previous action.",
    )
    parser.add_argument(
        "--in-embed",
        type=int,
        default=128,
        help="The dimensionality of the network. In the transformer, this is referred to as `d_model`.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=-1,
        help="The maximum number of steps allowed in the environment. If `env` has a `max_episode_steps`, this will be inferred. Otherwise, this argument must be supplied.",
    )
    parser.add_argument("--seed", type=int, default=42, help="The random seed to use.")
    parser.add_argument(
        "--save-policy",
        action="store_true",
        help="Use this to save the policy so you can load it later for rendering.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print out evaluation results as they come in to the console.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enjoy mode (NOTE: must have a trained policy saved).",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=50,
        help="This is how many (intermediate) Q-values we use to train for each context. To turn off intermediate Q-value prediction, set `--history 1`. To use the entire context, set history equal to the context length.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="Number of heads to use for the transformer.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of transformer blocks to use for the transformer.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout probability."
    )
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor.")
    parser.add_argument(
        "--gate",
        type=str,
        default="res",
        choices=["res", "gru"],
        help="Combine step to use.",
    )
    parser.add_argument(
        "--identity",
        action="store_true",
        help="Whether or not to use identity map reordering.",
    )
    parser.add_argument(
        "--pos",
        default="learned",
        choices=["learned", "sin", "none"],
        help="The type of positional encodings to use.",
    )
    parser.add_argument(
        "--bag-size", type=int, default=0, help="The size of the persistent memory bag."
    )
    return parser.parse_args()


def evaluate(
    agent: DtqnAgent,
    eval_env: gym.Env,
    eval_episodes: int
):
    agent.eval_on()

    total_reward = 0
    num_successes = 0
    total_steps = 0

    progress_bar = tqdm(range(1, 1 + eval_episodes))
    for i_episode in progress_bar:
        agent.context_reset(eval_env.reset()[0])
        done = False
        ep_reward = 0
        while not done:
            action = agent.get_action(epsilon=0.0)
            obs_next, reward, done, trunc, info = eval_env.step(action)
            done = done or trunc
            agent.observe(obs_next, action, reward, done)
            ep_reward += reward
            progress_bar.set_description(f"[Eval {i_episode}/{eval_episodes}] Ep. reward = {ep_reward}")
        total_reward += ep_reward
        total_steps += agent.context.timestep
        if ep_reward > 0:
            num_successes += 1

    # Set networks back to train mode
    agent.eval_off()
    # Prevent divide by 0
    episodes = max(eval_episodes, 1)
    return (
        num_successes / episodes,
        total_reward / episodes,
        total_steps / episodes,
    )


def train(
    agent: DtqnAgent,
    envs: Tuple[gym.Env],
    eval_envs: Tuple[gym.Env],
    total_steps: int,
    eps: EpsilonAnneal,
    eval_frequency: int,
    eval_episodes: int,
    policy_path: str,
    save_policy: bool,
    logger,
    mean_success_rate: RunningAverage,
    mean_episode_length: RunningAverage,
    mean_reward: RunningAverage,
    time_remaining: Optional[int],
    verbose: bool = False,
) -> None:

    start_time = time()
    agent.eval_off()
    env = RNG.rng.choice(envs)
    agent.context_reset(env.reset()[0])

    progress_bar = tqdm(range(agent.num_train_steps, total_steps))
    for timestep in progress_bar:
        done = step(agent, env, eps)

        if done:
            agent.replay_buffer.flush()
            env = RNG.rng.choice(envs)
            agent.context_reset(env.reset()[0])
        agent.train()
        eps.anneal()

        if timestep % eval_frequency == 0:
            hours = (time() - start_time) / 3600
            # Log training values
            log_vals = {
                "losses/TD_Error": agent.td_errors.mean(),
                "losses/Grad_Norm": agent.grad_norms.mean(),
                "losses/Max_Q_Value": agent.qvalue_max.mean(),
                "losses/Mean_Q_Value": agent.qvalue_mean.mean(),
                "losses/Min_Q_Value": agent.qvalue_min.mean(),
                "losses/Max_Target_Value": agent.target_max.mean(),
                "losses/Mean_Target_Value": agent.target_mean.mean(),
                "losses/Min_Target_Value": agent.target_min.mean(),
                "losses/hours": hours,
            }
            for i, eval_env in enumerate(eval_envs):
                sr, ret, length = evaluate(agent, eval_env, eval_episodes)

                log_vals.update(
                    {
                        f"{i}/SuccessRate": sr,
                        f"{i}/Return": ret,
                        f"{i}/EpisodeLength": length,
                    }
                )

            logger.log(
                log_vals,
                step=timestep,
            )

            if verbose:
                print(
                    f"[ {timestamp()} ] Training Steps: {timestep}, Success Rate: {sr:.2f}, Return: {ret:.2f}, Episode Length: {length:.2f}, Hours: {hours:.2f}"
                )

        if timestep % 50_000 == 0:
            torch.save(agent.policy_network.state_dict(), policy_path)



def step(agent: DtqnAgent, env: gym.Env, eps: float) -> bool:
    action = agent.get_action(epsilon=eps.val)
    next_obs, reward, done, trunc, info = env.step(action)
    done = done or trunc
    agent.observe(next_obs, action, reward, done)
    return done


def prepopulate(agent: DtqnAgent, prepop_steps: int, envs: Tuple[gym.Env]) -> None:
    timestep = 0
    with tqdm(total=prepop_steps, desc="Prepopulating") as progress_bar:
        while timestep < prepop_steps:
            env = RNG.rng.choice(envs)
            agent.context_reset(env.reset()[0])
            done = False
            while not done:
                action = RNG.rng.integers(env.action_space.n)
                next_obs, reward, done, trunc, info = env.step(action)
                done = done or trunc
                agent.observe(next_obs, action, reward, done)
                timestep += 1
                progress_bar.update(1)
            agent.replay_buffer.flush()


def run_experiment(args, envs: Tuple[gym.Env], eval_envs: Tuple[gym.Env]):
    start_time = time()
    device = torch.device(args.device)
    set_global_seed(args.seed, *(envs + eval_envs))

    eps = LinearAnneal(1.0, 0.1, args.num_steps // 10)

    agent = get_agent(
        envs,
        args.obs_embed,
        args.a_embed,
        args.in_embed,
        args.buf_size,
        device,
        args.lr,
        args.batch,
        args.context,
        args.max_episode_steps,
        args.history,
        args.tuf,
        args.discount,
        args.heads,
        args.layers,
        args.dropout,
        args.identity,
        args.gate,
        args.pos,
        args.bag_size,
    )

    print(
        f"[ {timestamp()} ] Creating DTQN model with {sum(p.numel() for p in agent.policy_network.parameters())} parameters"
    )

    # Create logging dir
    policy_save_dir = os.path.join(os.getcwd(), "dtqn", "policies")
    os.makedirs(policy_save_dir, exist_ok=True)
    policy_path = os.path.join(
        policy_save_dir,
        f"model=DTQN_envs=HIV_obs_embed={args.obs_embed}_a_embed={args.a_embed}_in_embed={args.in_embed}_context={args.context}_heads={args.heads}_layers={args.layers}_"
        f"batch={args.batch}_gate={args.gate}_identity={args.identity}_history={args.history}_pos={args.pos}_bag={args.bag_size}_seed={args.seed}",
    )

    # Enjoy mode
    if args.render:
        agent.policy_network.load_state_dict(
            torch.load(policy_path, map_location="cpu")
        )
        evaluate(agent, eval_envs[0], 1_000_000, render=True)

    # If there is already a saved checkpoint, load it and resume training if more steps are needed
    # Or exit early if we have already finished training.
    if os.path.exists(policy_path + "_mini_checkpoint.pt"):
        steps_completed = agent.load_mini_checkpoint(policy_path)["step"]
        print(
            f"Found a mini checkpoint that completed {steps_completed} training steps."
        )
        if steps_completed >= args.num_steps:
            print(f"Removing checkpoint and exiting...")
            if os.path.exists(policy_path + "_checkpoint.pt"):
                os.remove(policy_path + "_checkpoint.pt")
            exit(0)
        else:
            (
                mean_success_rate,
                mean_reward,
                mean_episode_length,
                eps_val,
            ) = agent.load_checkpoint(policy_path)
            eps.val = eps_val
    else:
        prepopulate(agent, 50_000, envs)
        mean_success_rate = RunningAverage(10)
        mean_reward = RunningAverage(10)
        mean_episode_length = RunningAverage(10)

    logger = get_logger(policy_path, args, envs)

    time_remaining = None

    train(
        agent,
        envs,
        eval_envs,
        args.num_steps,
        eps,
        args.eval_frequency,
        args.eval_episodes,
        policy_path,
        args.save_policy,
        logger,
        mean_success_rate,
        mean_reward,
        mean_episode_length,
        time_remaining,
        args.verbose,
    )

    agent.save_mini_checkpoint(checkpoint_dir=policy_path)


if __name__ == "__main__":
    run_experiment(get_args())
