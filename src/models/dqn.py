import matplotlib.pyplot as plt
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations=6, n_actions=5):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class Config:
    def __init__(self):
        self.batch_size = 128
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.tau = 0.005
        self.lr = 1e-4
        self.replay_memory_capacity = int(1e6)
        self.n_observations = 6
        self.n_actions = 4
        self.mse_loss = False
        self.num_episodes = 1_000


PATH = "dqn_agent.pt"


class DQNAgent:

    def __init__(self, config=Config()):
        self.config = config

        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.config.lr, amsgrad=True)
        self.criterion = nn.MSELoss() if self.config.mse_loss else nn.SmoothL1Loss()
        self.memory = ReplayMemory(capacity=self.config.replay_memory_capacity)

        self.steps_done = 0
        self._train = True

    def set_env(self, env):
        self.env = env

    def act(self, state, use_random=False):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state, device=device)
            if state.ndim == 1:
                state = state.unsqueeze(0)

        if not self._train:
            with torch.no_grad():
                return self.policy_net(state).argmax(1).view(1, 1)
        
        sample = random.random()
        eps_threshold = self.config.eps_end + (self.config.eps_start - self.config.eps_end) * \
            math.exp(-1. * self.steps_done / self.config.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).argmax(1).view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)

    def save(self, path=PATH):
        torch.save(self.policy_net.state_dict(), path)

    def load(self):
        self._train = False
        self.policy_net = DQN().to(device)
        self.policy_net.load_state_dict(torch.load(PATH))
        self.policy_net.eval()

    def optimize_model(self):
        if len(self.memory) < self.config.batch_size:
            return
        transitions = self.memory.sample(self.config.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.config.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.config.gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self):
        scores = []
        for i_episode in tqdm(range(self.config.num_episodes), "Training DQN"):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            episode_score = 0
            for t in count():
                action = self.act(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                episode_score += reward
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)
                state = next_state
                self.optimize_model()

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.config.tau + target_net_state_dict[key]*(1-self.config.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    break
            
            print(f"[Episode {i_episode + 1} / {self.config.num_episodes}] : Score = {episode_score}")
            scores.append(episode_score)

        plt.figure()
        plt.plot(scores)
        plt.show()
    