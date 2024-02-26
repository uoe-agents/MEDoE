from abc import ABC, abstractmethod
from typing import Dict, Union, Iterable
import gym
import torch
import os

AgentID = Union[int, str]


class Agent(ABC):
    def __init__(
            self,
            action_space: gym.Space,
            obs_space: gym.Space
            ):
        self.action_space = action_space
        self.obs_space = obs_space
        self.saveables = {}

    def save_params(self, pathname: str):
        torch.save(self.saveables, pathname)

    def load_params(self, pathname: str):
        pathname = os.path.join(os.path.dirname(__file__), pathname)

        load_params = torch.load(pathname)
        for k, v in self.saveables.items():
            v.load_state_dict(load_params[k].state_dict())

    @abstractmethod
    def act(self):
        ...

    @abstractmethod
    def update(self):
        ...

    @abstractmethod
    def store_transition(self, transition):
        ...

    @abstractmethod
    def new_episode(self):
        "Resets any relevant variables at the start of an episode"
        ...


class MultiAgent(Agent):
    def __init__(
            self,
            ids: Iterable[AgentID],
            action_spaces: Dict[AgentID, gym.Space],
            obs_spaces: Dict[AgentID, gym.Space],
            joint_space: gym.Space,
            ):
        """
        MultiAgent

        Params:
            action_spaces -- action spaces, one for each of the agents
            obs_space -- observation spaces, one for each of the agents
            joint_space -- a joint observation space for centralised training
            n_agents -- number of agents
        """
        self.ids = ids
        self.action_spaces = action_spaces
        self.obs_spaces = obs_spaces
        self.joint_space = joint_space
        self.n_agents = len(ids)
        self.saveables = {}
        self.agents = {}

    def __getitem__(self, key):
        return self.agents[key]

    def add_agent(
            self,
            agent_id: int,
            action_space: gym.Space,
            obs_space: gym.Space,
            ):
        assert agent_id not in self.ids, \
                f"Agent {agent_id} already exists."
        self.ids.append(agent_id)
        self.action_spaces[agent_id] = action_space
        self.obs_spaces[agent_id] = obs_space
        # TODO need to update joint space

    def del_agent(self, agent_id: int):
        try:
            del self.action_spaces[agent_id]
            del self.obs_spaces[agent_id]
            self.ids.remove(agent_id)
        except KeyError:
            pass
