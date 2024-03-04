from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import copy
import numpy as np
import torch
import warnings
warnings.filterwarnings(action="ignore")

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

from models import dtqn


class ProjectAgent(object):

    def __init__(self):
        self.dtqn_agent = dtqn.get_agent(
            envs=[env],
            embed_per_obs_dim=8,
            action_dim=0,
            inner_embed=128,
            buffer_size=500_000,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            learning_rate=3e-4,
            batch_size=32,
            context_len=50,
            max_env_steps=-1,
            history=50,
            target_update_frequency=10_000,
            gamma=0.99,
            num_heads=8,
            num_layers=2,
            dropout=0.0,
            identity=False,
            gate="res",
            pos="learned",
            bag_size=0,
        )
        dtqn.set_global_seed(42)

        self.dtqn_agent.eval_on()
        self.env = copy.deepcopy(env) # reward estimator
        self.last_observation = None
        self.last_action = None
        self.steps = 0
        self.policy_path = "dtqn.pt"

    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
        done = (self.steps == 200)

        if self.steps == 200:
            self.steps = 0
            self.last_observation = None
            self.last_action = None

        if self.last_observation is None or self.last_action is None:
            self.dtqn_agent.context_reset(observation)

        action = self.dtqn_agent.get_action(epsilon=0.0)
        self.steps += 1

        if self.last_observation is not None and self.last_action is not None:
            reward = self.env.reward(
                self.last_observation, 
                self.env.action_set[self.last_action], 
                observation
            ) # estimator
            self.dtqn_agent.observe(observation, self.last_action, reward, done)

        self.last_observation = observation
        self.last_action = action
        return action

    def save(self, path: str) -> None:
        torch.save(self.dtqn_agent.policy_network.state_dict(), path)

    def load(self) -> None:
        self.dtqn_agent.policy_network.load_state_dict(
            torch.load(self.policy_path, map_location="cpu")
        )
        
        


if __name__ == "__main__":
    dtqn.run_experiment(
        dtqn.get_args(),
        [env],
        [env]
    )
