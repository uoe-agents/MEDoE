import functools
from copy import copy
from enum import Enum
from gym.utils import seeding
from gym.spaces import Discrete, Box, MultiBinary
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec

def env(loader="simple", **kwargs):
    ret_env = raw_env(loader=loader, **kwargs)
    ret_env = wrappers.AssertOutOfBoundsWrapper(ret_env)
    ret_env = wrappers.OrderEnforcingWrapper(ret_env)
    return ret_env

def raw_env(loader="simple", **kwargs):
    ret_env = parallel_env(loader=loader, **kwargs)
    ret_env = parallel_to_aec(ret_env)
    return ret_env

def parallel_env(
        loader="simple",
        **kwargs
        ):
    if loader == "simple":
        return ChainballEnv(**kwargs)
    elif loader == "config":
        return ChainballEnv.from_config(
            kwargs["cfg"]
            )
    else:
        raise Exception(f"Environment loader {loader} not known for CorridorEnv")

class ChainballEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "ansi"],
        "render_fps": 2,
        "name": "chainball_v0"
    }

    def __init__(self, 
                 max_cycles=np.Inf,
                 num_agents=2,
                 num_actions=4,
                 chain_length=7,
                 spawn_location=4,
                 fixed_time=False,
                 terminal_states=[],
                 optimals=None,
                 enable_our_goal=True,
                 enable_opp_goal=True,
                 render_mode="human",
                 obs_mode="onehot",
                 normalize_reward=True,
                 beta_a=1,
                 beta_b=1,
                 optimal_level=0.8,
                 noise_level=0.5,
                 default_back_coef=1.5,
                 env_gen_seed=1,
                 ):

        self.chain_length = chain_length
        self.num_actions = num_actions
        self.max_cycles = max_cycles
        self.possible_agents = [f"player_{i}" for i in range(num_agents)]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}
        self.spawn_location = spawn_location
        self.terminal_states = terminal_states
        self.enable_our_goal = enable_our_goal
        self.enable_opp_goal = enable_opp_goal
        self.fixed_time = fixed_time
        self.obs_mode = obs_mode
        self.render_mode = render_mode
        self.normalize_reward = normalize_reward
        self.beta_a=beta_a
        self.beta_b=beta_b
        self.optimal_level=optimal_level
        self.noise_level=noise_level
        self.default_back_coef=default_back_coef
        if self.fixed_time:
            assert np.isfinite(max_cycles), "Expect max_cycles to be finite when using fixed_time=True"
        self.loc = self.spawn_location
        self.our_goal_loc = 0
        self.opp_goal_loc = self.chain_length + 1
        self.agents = []
        if optimals is None:
            self.optimals = [(0 for _ in range(self.num_agents))
                             for _ in range(self.chain_length+2)]
        else:
            self.optimals=optimals

        self.env_gen_random, _ = seeding.np_random(env_gen_seed)
        self.create_transition_matrix(
            self.optimals,
            beta_a=self.beta_a,
            beta_b=self.beta_b,
            optimal_level=self.optimal_level,
            noise_level=self.noise_level,
            default_back_coef=self.default_back_coef
            )

    @classmethod
    def from_config(cls, cfg):
        return cls(
                 max_cycles=cfg.get("max_cycles", np.Inf),
                 num_agents=cfg.get("num_agents", 2),
                 num_actions=cfg.get("num_actions", 4),
                 chain_length=cfg.get("chain_length", 7),
                 spawn_location=cfg.get("spawn_location", 4),
                 fixed_time=cfg.get("fixed_time", False),
                 terminal_states=cfg.get("terminal_states", []),
                 optimals=cfg.get("optimals", None),
                 enable_our_goal=cfg.get("enable_our_goal", True),
                 enable_opp_goal=cfg.get("enable_opp_goal", True),
                 render_mode=cfg.get("render_mode", "human"),
                 obs_mode=cfg.get("obs_mode", "onehot"),
                 normalize_reward=cfg.get("normalize_reward", True),
                 beta_a=cfg.get("beta_a", 1),
                 beta_b=cfg.get("beta_b", 1),
                 optimal_level=cfg.get("optimal_level", 0.8),
                 noise_level=cfg.get("noise_level", 0.5),
                 default_back_coef=cfg.get("default_back_coef", 1.5),
                 env_gen_seed=1,
            )

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if self.obs_mode == "discrete":
            return Discrete(2+self.chain_length)
        if self.obs_mode == "index_box":
            return Box(low=0, high=2+self.chain_length, shape=(1,), dtype=np.float32)
        if self.obs_mode == "onehot":
            return Box(low=0, high=1, shape=(2+self.chain_length,), dtype=np.float32)
        if self.obs_mode == "onehot_binary":
            return MultiBinary(2+self.chain_length)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.num_actions)

    def render(self, mode=None):
        #    □.o.....□       □.......▣
        #      normal          score
        if mode is None:
            mode = self.render_mode
        base = list("□" + "."*self.chain_length + "□")
        if self.loc in (self.our_goal_loc, self.opp_goal_loc):
            base[self.loc] = "▣"
        else:
            base[self.loc] = "o"
        if mode == "ansi":
            return "".join(base)
        else:
            print("".join(base))

    def close(self):
        pass

    def reset(self, seed=None, return_info=False, options=None):
        self.seed(seed)
        self.agents = copy(self.possible_agents)
        self.loc = self.spawn_location
        self.current_step = 0
        observations = {agent: self.loc_obs(self.loc)
                        for agent in self.agents}
        if return_info:
            infos = {agent: {} for agent in self.agents}
            return observations, infos
        return observations

    def loc_obs(self, loc):
        if self.obs_mode == "discrete":
            return loc
        if self.obs_mode == "index_box":
            return np.array([loc], dtype=np.float32)
        if self.obs_mode == "onehot":
            return np.array([(i==loc) for i in range(2+self.chain_length)], dtype=np.float32)
        if self.obs_mode == "onehot_binary":
            return np.array([(i==loc) for i in range(2+self.chain_length)], dtype=np.int8)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def next_loc(self, actions):
        # handle case when in a state from which we should respawn:
        if self.loc in (self.our_goal_loc, self.opp_goal_loc):
            return self.spawn_location
        next_loc = self.sample_transition(self.loc, actions)
        if next_loc in self.terminal_states:
            return -1
        return next_loc

    def compute_max_return(self, n_eps):
        total_return = 0
        for _ in range(n_eps):
            self.reset()
            done = False
            while not done:
                opt_action = np.unravel_index(
                        np.argmax(self.forward_prob_matrix[self.loc]),
                        self.forward_prob_matrix[self.loc].shape
                        )
                _, r, te, tr, _ = self.step(dict(zip(self.agents, opt_action)))
                done = te["player_0"] or tr["player_0"]
                total_return += r["player_0"]
        return len(self.possible_agents)*total_return/n_eps

    def compute_min_return(self, n_eps):
        total_return = 0
        for _ in range(n_eps):
            self.reset()
            done = False
            while not done:
                opt_action = np.unravel_index(
                        np.argmin(self.forward_prob_matrix[self.loc]),
                        self.forward_prob_matrix[self.loc].shape
                        )
                _, r, te, tr, _ = self.step(dict(zip(self.agents, opt_action)))
                done = te["player_0"] or tr["player_0"]
                total_return += r["player_0"]
        return 4*total_return/n_eps

    def compute_reward_normalization(self, n_eps):
        max = self.compute_max_return(n_eps)
        min = self.compute_min_return(n_eps)
        self.reward_transform = lambda x: (x-min)/(max-min)

    def step(self, actions):
        self.loc = self.next_loc(actions)
        in_terminal_state = (self.loc == -1)
        self.current_step += 1
        observations = {agent: self.loc_obs(self.loc)
                        for agent in self.agents}
        if (self.loc == self.our_goal_loc) and self.enable_our_goal:
            reward = -1
        elif (self.loc == self.opp_goal_loc) and self.enable_opp_goal:
            reward = 1
        else:
            reward = 0
        rewards = {agent: reward for agent in self.agents}
        if self.fixed_time:
            terminations = {agent: in_terminal_state or (self.current_step >= self.max_cycles)
                            for agent in self.agents}
            truncations = {agent: self.current_step >= self.max_cycles
                           for agent in self.agents}
        else:
            terminations = {agent: in_terminal_state or (reward != 0)
                            for agent in self.agents}
            truncations = {agent: self.current_step >= self.max_cycles
                           for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        self.agents = [agent
                       for agent in self.agents
                       if not (terminations[agent] or truncations[agent])]
        return observations, rewards, terminations, truncations, infos

    def sample_transition(self, loc, actions):
        action_list = [actions[agent] for agent in self.agents]
        tm_index = (loc, *action_list)

        forward_prob = self.forward_prob_matrix[tm_index]
        backward_coef = self.backward_coef_matrix[tm_index]

        if self.np_random.random() < forward_prob:
            return loc + 1
        else:
            p = np.array([backward_coef**i for i in range(loc)])
            p = p/p.sum()
            return self.np_random.choice(loc, p=p)

    def create_transition_matrix(self, optimals, beta_a=1, beta_b=1, optimal_level=0.8, noise_level=0.5, default_back_coef=1.5):
        # A more 'football-like' game would have the forward probability depend strongly on attackers actions in attack states, and the decay prob depend on the defenders
        # e.g.
        # row players: defenders
        # (imagine these normalised)
        #
        #                def    mid    att
        # forward_prob  [3 3]  [2 3]  [1 3]
        #               [1 1]  [1 2]  [1 3]
        #
        # decay_coef    [2 2]  [2 2]  [2 2]
        #               [2 2]  [3 3]  [4 4]
        #
        # I will leave this as a future extension to the environment

        # define by providing a parsed list of optimal joint actions,
        # e.g. with 2 actions {0,1}:
        # [(1,0,None), (1,None,0)] would have:
        # state 1 optimal: (1,0,0), (1,0,1)
        # state 2 optimal: (1,0,0), (1,1,0)
        # we might also want to make it such that agents who do the wrong action have different levels of 'impact'
        # e.g. we might want (None,0,0) to range between 0.25 and 0.5 (so perhaps not an 'optimal' forward prob like 0.75, but not badly suboptimal eitherj)
        matrix_size = [self.chain_length + 2] + len(self.possible_agents)*[self.num_actions]
        forward_prob_matrix = noise_level * self.env_gen_random.beta(beta_a, beta_b, size=matrix_size)
        for state, optimal in enumerate(optimals):
            action_slice = [(slice(None) if ((a is None) or (a<0)) else a)
                            for a in optimal]
            forward_prob_matrix[(state, *action_slice)] = optimal_level
        self.forward_prob_matrix = forward_prob_matrix
        self.backward_coef_matrix = default_back_coef * np.ones(matrix_size)
