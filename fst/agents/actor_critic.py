import os
import os.path as osp
from collections import defaultdict
import gym
import numpy as np
import torch
from torch.optim import Adam
from typing import Iterable
from fst.utils.experience import ReplayBuffer, Transition
from fst.agents.agents import Agent, MultiAgent
from fst.utils.network import fc_network
from copy import copy, deepcopy
from omegaconf import OmegaConf
import hydra


class A2C(Agent):
    def __init__(
            self,
            action_space: gym.Space,
            obs_space: gym.Space,
            actor_hidden_size: Iterable[int],
            critic_hidden_size: Iterable[int],
            actor_learning_rate: float,
            critic_learning_rate: float,
            gamma: float,
            ent_coef: float,
            n_steps: int,
            clip_grad={},
            save_to_zoo=False,
            freeze_critic=0,
            freeze_actor=0,
            n_train_envs=1,
            ignore_trunc=True,
            **kwargs
        ):
        super().__init__(action_space, obs_space)

        state_size = obs_space.shape[0]
        action_size = action_space.n
        self.action_size = action_size

        # Initialise the Actor
        self.actor_net = fc_network([
            state_size,
            *actor_hidden_size,
            action_size
            ])
        self.actor_optim = Adam(
            self.actor_net.parameters(),
            lr=actor_learning_rate,
            eps=1e-5
            )
        self.actor_learning_rate = actor_learning_rate

        # Initialise the Critic
        self.critic_net = fc_network([
            state_size,
            *critic_hidden_size,
            1
            ])
        self.critic_optim = Adam(
            self.critic_net.parameters(),
            lr=critic_learning_rate,
            eps=1e-5
            )
        self.critic_learning_rate = critic_learning_rate

        # Initialise hyperparameters
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.n_steps = n_steps
        self.ignore_trunc = ignore_trunc
        self.clip_grad = defaultdict(lambda: False)
        self.clip_grad.update(clip_grad)
        self.n_train_envs = n_train_envs
        self.freeze_critic = freeze_critic/self.n_train_envs
        self.freeze_actor = freeze_actor/self.n_train_envs

        # Initialise update counters
        self.c_update = 0
        self.p_update = self.n_steps
        self.total_critic_updates = 0
        self.total_actor_updates = 0

        # Initialise memory
        self.last_transition = None
        self.last_n_transitions = ReplayBuffer(n_steps)

        self.saveables.update({
            "critic": self.critic_net,
            "actor": self.actor_net,
            "critic_optim": self.critic_optim,
            "actor_optim": self.actor_optim
            })
        self.save_to_zoo = save_to_zoo

    def policy(self, obs, explore=True):
        with torch.no_grad():
            logits = self.actor_net(torch.from_numpy(obs).float())
            dist = torch.distributions.categorical.Categorical(logits=logits)
        return dist

    def act(self, obs, explore=True):
        policy = self.policy(obs=obs,
                             explore=explore
                             )
        return policy.sample().unsqueeze(1).numpy()
        #return policy.sample().numpy()

    def update(self):
        self.c_update += 1
        critic_frozen = self.c_update <= self.freeze_critic
        actor_frozen = self.c_update <= self.freeze_actor
        if (critic_frozen and actor_frozen) or self.c_update % self.p_update != 0:
            return {}
        # generate G, V, lp_chosen, entropy
        obs = torch.stack(list(torch.from_numpy(t.obs)
                               for t in self.last_n_transitions)).float()
        n_obs = torch.from_numpy(self.last_transition.n_obs).float()
        actions = torch.stack(
                      list(torch.from_numpy(t.action)
                           for t in
                           self.last_n_transitions))
        V = self.critic_net(obs)
        G = torch.zeros_like(V)
        # imp_weights = torch.ones_like(V)
        for i, t in enumerate(reversed(self.last_n_transitions)):
            # return
            d = torch.from_numpy(t.terminated).unsqueeze(1)
            trunc = torch.from_numpy(t.truncated).unsqueeze(1)
            r = torch.from_numpy(t.reward).unsqueeze(1)
            n = self.critic_net(n_obs) if i == 0 else G[-i, :, :]
            if self.ignore_trunc:
                G[-i-1, :, :] = (
                    r + self.gamma * ~(d|trunc) * n
                    )
            else:
                G[-i-1, :, :] = (
                    ~trunc * (r + self.gamma * ~d * n)
                    + trunc * V[-i-1, :, :]
                    )
        # cum_weight = 1
        # for i, t in enumerate(self.last_n_transitions):
        #     # importance weight
        #     d = torch.from_numpy(t.done).unsqueeze(1)
        #     imp_weights[i, :, :] = cum_weight * t.imp_weight
        #     cum_weight = imp_weights[i,:,:] ** ~d
        A = (G-V).detach()
        logits = self.actor_net(obs)
        lp = torch.nn.functional.log_softmax(logits, dim=2)
        lp_chosen = lp.gather(2, actions)
        entropy = -lp.exp().mul(lp).sum(dim=2).unsqueeze(2)
        # update
        critic_metrics = self._update_critic(G, V, frozen=critic_frozen)
        if not critic_frozen:
            self.total_critic_updates += 1
        actor_metrics = self._update_actor(A, lp_chosen, entropy, frozen=actor_frozen)
        if not actor_frozen:
            self.total_actor_updates += 1
        # return
        train_metrics = {
            "nTD": A.mean().item(),
            # "imp_weights": imp_weights.mean().item(),
            **critic_metrics,
            **actor_metrics
            }
        return train_metrics

    def _update_critic(self, G, V, frozen=False):
        if frozen:
            with torch.no_grad():
                critic_loss = torch.nn.functional.mse_loss(G, V)
        else:
            self.critic_optim.zero_grad()
            critic_loss = torch.nn.functional.mse_loss(G, V)
            critic_loss.backward()
            if self.clip_grad.get("critic"):
                torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.clip_grad["critic"])
            self.critic_optim.step()
        critic_metrics = {
                "critic_loss": critic_loss.item()
                }
        if not frozen:
            for i, p in enumerate(self.critic_net.parameters()):
                critic_metrics[f"critic_grad_{i}"] = p.grad.detach().norm().item()
        return critic_metrics

    def _update_actor(self, A, lp_chosen, entropy, weights=1.0, frozen=False):
        if frozen:
            with torch.no_grad():
                actor_loss = - (lp_chosen.mul(A)
                                + self.ent_coef*entropy).mul(weights).mean()
        else:
            self.actor_optim.zero_grad()
            actor_loss = - (lp_chosen.mul(A)
                            + self.ent_coef*entropy).mul(weights).mean()
            actor_loss.backward()
            if self.clip_grad.get("actor"):
                torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.clip_grad["actor"])
            self.actor_optim.step()
        actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy.mean().item(),
                }
        if not frozen:
            for i, p in enumerate(self.actor_net.parameters()):
                actor_metrics[f"actor_grad_{i}"] = p.grad.detach().norm().item()
        return actor_metrics

    def store_transition(self, transition):
        self.last_transition = transition
        self.last_n_transitions.push(transition)

    @property
    def info(self):
        i = {
            "type": "A2C",
            "gamma": self.gamma,
            "ent_coef": self.ent_coef,
            "n_steps": self.n_steps,
            "lr": {"actor": self.actor_learning_rate,
                   "critic": self.critic_learning_rate},
            "train_steps": self.c_update * self.n_train_envs,
            "critic_updates": self.total_critic_updates,
            "actor_updates": self.total_actor_updates,
            }
        return i




class IA2C(MultiAgent):
    def __init__(
            self,
            ids=[],
            action_spaces={},
            obs_spaces={},
            joint_space=None,
            actor_hidden_sizes={},
            critic_hidden_sizes={},
            actor_learning_rates={},
            critic_learning_rates={},
            gamma=1.0,
            ent_coef=0.0,
            clip_grad={},
            n_steps=5,
            exp_buffer_size=0,
            exp_buffer_replacement_prob=1.0,
            save_to_zoo=False,
            freeze_critic=0,
            freeze_actor=0,
            n_train_envs=1,
            ignore_trunc=True,
        ):
        super().__init__(
            ids,
            action_spaces,
            obs_spaces,
            joint_space
            )
        state_size = list(obs_spaces.values())[0].shape[0]
        action_size = list(action_spaces.values())[0].n
        self.action_size = action_size

        params = []

        # Initialise the decentralised actors & critics
        self.actor_nets = {}
        self.actor_learning_rates = {}
        self.critic_nets = {}
        self.critic_learning_rates = {}
        for agent_id in self.ids:
            # Actors
            self.actor_nets[agent_id] = fc_network([
                state_size,
                *actor_hidden_sizes[agent_id],
                action_size
                ])
            self.actor_learning_rates[agent_id] = actor_learning_rates[agent_id]
            params.append({
                "params": self.actor_nets[agent_id].parameters(),
                "lr": self.actor_learning_rates[agent_id],
                })
            # Critics
            self.critic_nets[agent_id] = fc_network([
                state_size,
                *critic_hidden_sizes[agent_id],
                1
                ])
            self.critic_learning_rates[agent_id] = critic_learning_rates[agent_id]
            params.append({
                "params": self.critic_nets[agent_id].parameters(),
                "lr": self.critic_learning_rates[agent_id],
                })

        self.optim = Adam(params,
                          lr=params[0]["lr"],
                          eps=1e-5)

        # Initialise hyperparameters
        self.gamma = gamma
        self.gae_lambda = 1.0
        self.ent_coef = ent_coef
        self.n_steps = n_steps
        self.ignore_trunc = ignore_trunc
        self.clip_grad = defaultdict(lambda: False) 
        self.clip_grad.update(clip_grad)
        self.n_train_envs = n_train_envs
        self.freeze_critic = freeze_critic/self.n_train_envs
        self.freeze_actor = freeze_actor/self.n_train_envs
        # divide by n_train_envs since each call to update has n_train_envs transitions

        # Initialise update counters
        self.c_update = 0
        self.p_update = self.n_steps
        self.total_critic_updates = 0
        self.total_actor_updates = 0

        # Initialise memory
        self.last_transition = None
        self.last_n_transitions = ReplayBuffer(n_steps)
        self.exp_buffer_size = exp_buffer_size
        self.exp_buffer_replacement_prob = exp_buffer_replacement_prob
        self.exp_buffer = {agent_id: [] for agent_id in self.ids}

        self.saveables.update({
            "critic": self.critic_nets,
            "actor": self.actor_nets,
            "optim": self.optim,
            })
        self.save_to_zoo = save_to_zoo

    def policy(self, obs, explore=True, agent_id=None):
        with torch.no_grad():
            if agent_id is None:
                p = {agent_id: self.policy(obs, explore=explore, agent_id=agent_id)
                     for agent_id in self.ids}
            else:
                logits = self.actor_nets[agent_id](torch.Tensor(obs[agent_id]))
                p = torch.distributions.categorical.Categorical(logits=logits)
        return p

    def act(self, obs, explore=True, agent_id=None):
        if agent_id is None:
            return {agent_id: self.act(obs, explore=explore, agent_id=agent_id)
                    for agent_id in self.ids}
        else:
            policy = self.policy(obs, explore=explore, agent_id=agent_id)
            return policy.sample().unsqueeze(1).numpy()

    def update(self):
        self.c_update += 1
        critic_frozen = self.c_update <= self.freeze_critic
        actor_frozen = self.c_update <= self.freeze_actor
        if (critic_frozen and actor_frozen) or self.c_update % self.p_update != 0:
            return {}
        # generate G, V, lp_chosen, entropy
        _, agent_obs, actions = self._stack_transitions()
        As, Gs, Vs = self._compute_gae(agent_obs)
        lp_chosens, entropies = self._compute_policy_qty(agent_obs, actions)
        # update
        self.optim.zero_grad()
        critic_loss, critic_metrics = self._update_critic(Gs, Vs, frozen=critic_frozen)
        actor_loss, actor_metrics = self._update_actor(As, lp_chosens, entropies, frozen=actor_frozen)
        total_loss = critic_loss + actor_loss
        total_loss.backward()
        if not critic_frozen:
            self.total_critic_updates += 1
            for agent_id in self.ids:
                for i, p in enumerate(self.critic_nets[agent_id].parameters()):
                    critic_metrics[agent_id][f"critic_grad_{i}"] = p.grad.detach().norm().item()
        if not actor_frozen:
            self.total_actor_updates += 1
            for agent_id in self.ids:
                for i, p in enumerate(self.actor_nets[agent_id].parameters()):
                    actor_metrics[agent_id][f"actor_grad_{i}"] = p.grad.detach().norm().item()
        self.optim.step()

        # return
        train_metrics = {
            **critic_metrics,
            **actor_metrics
            }
        return train_metrics

    def _compute_gae(self, obs, agent_id=None):
        if agent_id is None:
            As, Gs, Vs = zip(*(self._compute_gae(obs, agent_id=agent_id)
                           for agent_id in self.ids))
            As = dict(zip(self.ids, As))
            Gs = dict(zip(self.ids, Gs))
            Vs = dict(zip(self.ids, Vs))
            return As, Gs, Vs
        else:
            V = self.critic_nets[agent_id](obs[agent_id])
            with torch.no_grad():
                G = torch.zeros_like(V)
                A = torch.zeros_like(V)
                V_n = self.critic_nets[agent_id](self.last_transition.n_obs[agent_id])
                G_n = V_n.clone()
                A_n = 0
                for i, t in enumerate(reversed(self.last_n_transitions)):
                    term = t.terminated.unsqueeze(1)
                    trunc = t.truncated.unsqueeze(1)
                    r = t.reward.unsqueeze(1)
                    if self.ignore_trunc:
                        V_n = ~(term|trunc) * V_n
                        G_n = ~(term|trunc) * G_n
                    else:
                        V_n = (~trunc * (V_n * ~term)
                               + trunc * (V_n))
                        G_n = (~trunc * (G_n * ~term)
                               + trunc * (G_n))
                    d = r + self.gamma * V_n - V[-i-1, :, :]
                    A[-i-1, :, :] = d + self.gamma * self.gae_lambda * A_n
                    G[-i-1, :, :] = r + self.gamma * G_n
                    A_n = A[-i-1, :, :]
                    G_n = G[-i-1, :, :]
                    V_n = V[-i-1, :, :]
            return A, G, V

    def _compute_policy_qty(self, obs, acts, agent_id=None):
        if agent_id is None:
            lp_chosens, entropies = zip(*(self._compute_policy_qty(obs, acts, agent_id)
                                          for agent_id in self.ids))
            lp_chosens = dict(zip(self.ids, lp_chosens))
            entropies = dict(zip(self.ids, entropies))
            return lp_chosens, entropies
        else:
            logits = self.actor_nets[agent_id](obs[agent_id])
            lp = torch.nn.functional.log_softmax(logits, dim=2)
            lp_chosen = lp.gather(2, acts[agent_id])
            entropy = -lp.exp().mul(lp).sum(dim=2).unsqueeze(2)
            return lp_chosen, entropy

    def _update_critic(self, G, V, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            critic_metrics = {}
            for agent_id in self.ids:
                critic_loss, critic_metrics[agent_id] = self._update_critic(G[agent_id], V[agent_id],
                                                                            agent_id=agent_id,
                                                                            frozen=frozen)
                total_loss += critic_loss
            return total_loss, critic_metrics
        else:
            if frozen:
                with torch.no_grad():
                    critic_loss = torch.nn.functional.mse_loss(G, V)
            else:
                critic_loss = torch.nn.functional.mse_loss(G, V)
                if self.clip_grad.get("critic"):
                    torch.nn.utils.clip_grad_norm_(self.critic_nets[agent_id].parameters(), self.clip_grad["critic"])
            critic_metrics = {
                "critic_loss": critic_loss.item()
                }
            return critic_loss, critic_metrics

    def _update_actor(self, A, lp_chosen, entropy,
                      weights=1.0, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A[agent_id],
                                                                         lp_chosen,
                                                                         entropy,
                                                                         weights,
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            if frozen:
                with torch.no_grad():
                    actor_loss = - (lp_chosen[agent_id].mul(A)
                                    + self.ent_coef*entropy[agent_id]).mul(weights).mean()
            else:
                actor_loss = - (lp_chosen[agent_id].mul(A)
                                + self.ent_coef*entropy[agent_id]).mul(weights).mean()
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_nets[agent_id].parameters(), self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy[agent_id].mean().item(),
                }
            return actor_loss, actor_metrics

    def store_transition(self, transitions, agent_id=None):
        # I expect this to come in as a dict of individual agent Transitions
        obs = {agent_id: torch.FloatTensor(t.obs)
               for agent_id, t in transitions.items()}
        n_obs = {agent_id: torch.FloatTensor(t.n_obs)
               for agent_id, t in transitions.items()}
        action = {agent_id: torch.LongTensor(t.action)
                  for agent_id, t in transitions.items()}
        # Assumes common-payoff
        reward = torch.FloatTensor(list(transitions.values())[0].reward)
        terminated = torch.BoolTensor(list(transitions.values())[0].terminated)
        truncated = torch.BoolTensor(list(transitions.values())[0].truncated)
        joint_obs = torch.FloatTensor(list(transitions.values())[0].obs)
        transition = Transition(
            obs=obs,
            action=action,
            n_obs=n_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            joint_obs=joint_obs,
            )
        self.last_transition = transition
        self.last_n_transitions.push(transition)
        self.store_exp(obs)

    def _stack_transitions(self):
        joint_obs = torch.stack(list(t.joint_obs for t in self.last_n_transitions))
        agent_obs = {agent_id: torch.stack(list(t.obs[agent_id] for t in self.last_n_transitions))
                     for agent_id in self.ids}
        actions = {agent_id: torch.stack(list(t.action[agent_id] for t in self.last_n_transitions))
                   for agent_id in self.ids}
        return joint_obs, agent_obs, actions

    def store_exp(self, obs, agent_id=None):
        if agent_id is None:
            for agent_id in self.ids:
                self.store_exp(obs, agent_id=agent_id)
        else:
            if self.exp_buffer_size == 0:
                pass
            elif len(self.exp_buffer[agent_id]) >= self.exp_buffer_size:
                if np.random.random() < self.exp_buffer_replacement_prob:
                    idx = np.random.randint(self.exp_buffer_size)
                    self.exp_buffer[idx] = obs[agent_id]
            else:
                self.exp_buffer[agent_id].append(obs[agent_id])

    def add_agent(
            self,
            agent_id,
            action_space: gym.Space,
            obs_space: gym.Space,
            actor_hidden_size,
            critic_hidden_size,
            actor_learning_rate,
            critic_learning_rate,
            gamma,
            ent_coef,
            n_steps,
            save_to_zoo=False
            ):
        ...

    @property
    def info(self):
        return dict(
            type="IA2C",
            train_steps=self.c_update*self.n_train_envs,
            actor_train_updates=self.total_actor_updates,
            critic_train_updates=self.total_critic_updates,
            actor_learning_rates=self.actor_learning_rates,
            critic_learning_rates=self.critic_learning_rates,
            gamma=self.gamma,
            ent_coef=self.ent_coef,
            clip_grad={"actor": self.clip_grad["actor"], "critic": self.clip_grad["critic"]},
            n_steps=self.n_steps,
            exp_buffer_size=self.exp_buffer_size,
            exp_buffer_replacement_prob=self.exp_buffer_replacement_prob,
            ignore_trunc=self.ignore_trunc
            )

    def zoo_save(self, zoo_path, agent_save_names, env_cfg, evals, seed):
        agent_save_name_mapping = dict(zip(self.ids, agent_save_names))
        concat_name = "_".join(agent_save_names)
        # configs
        # actors
        # inelegant way of changing the evals into floats (from numpy.floats)
        # required so omegaconf can read
        float_evals = {
                "team": {k: v.item() for k, v in evals["team"].items()},
                "individual": {k: {k: v.item() for k, v in v.items()}
                               for k, v in evals["individual"].items()},
                }
        info = self.info
        for agent_id in self.ids:
            agent_name = agent_save_name_mapping[agent_id]
            partner_agents = [agent_save_name_mapping[other_agent_id]
                              for other_agent_id in self.ids
                              if other_agent_id != agent_id]
            # Actor
            actor_model_pathname = osp.normpath(
                osp.join(zoo_path, "outputs", "actors", f"{agent_name}.pt")
                )
            actor_config_pathname = osp.normpath(
                osp.join(zoo_path, "configs", "actors", f"{agent_name}.yaml")
                )
            os.makedirs(osp.dirname(actor_model_pathname), exist_ok=True)
            os.makedirs(osp.dirname(actor_config_pathname), exist_ok=True)
            torch.save(self.actor_nets[agent_id], actor_model_pathname)
            actor_model_dict = dict(
                internal_name=agent_id,
                train_steps=info["train_steps"],
                train_updates=info["actor_train_updates"],
                learning_rate=info["actor_learning_rates"][agent_id],
                gamma=self.gamma,
                ent_coef=self.ent_coef,
                clip_grad=self.clip_grad["actor"],
                n_steps=self.n_steps,
                ignore_trunc=self.ignore_trunc,
                )
            actor_dict = dict(
                name=agent_name,
                model=actor_model_dict,
                train_env=env_cfg,
                eval={
                    "team": float_evals["team"],
                    "individual": float_evals["individual"][agent_id],
                    },
                critic=agent_name,
                experience=agent_name,
                partner_agents=partner_agents,
                path_to_model=actor_model_pathname,
                seed=seed,
                )
            actor_dict = OmegaConf.create(actor_dict)
            with open(actor_config_pathname, "w") as f:
                f.write(OmegaConf.to_yaml(actor_dict))

            # Critic
            critic_model_pathname = osp.normpath(
                osp.join(zoo_path, "outputs", "critics", f"{agent_name}.pt")
                )
            critic_config_pathname = osp.normpath(
                osp.join(zoo_path, "configs", "critics", f"{agent_name}.yaml")
                )
            os.makedirs(osp.dirname(critic_model_pathname), exist_ok=True)
            os.makedirs(osp.dirname(critic_config_pathname), exist_ok=True)
            torch.save(self.critic_nets[agent_id], critic_model_pathname)
            critic_model_dict = dict(
                internal_name=agent_id,
                train_steps=info["train_steps"],
                train_updates=info["critic_train_updates"],
                learning_rate=info["critic_learning_rates"][agent_id],
                gamma=self.gamma,
                ent_coef=self.ent_coef,
                clip_grad=self.clip_grad["critic"],
                n_steps=self.n_steps,
                ignore_trunc=self.ignore_trunc,
                )
            critic_dict = dict(
                name=agent_name,
                model=critic_model_dict,
                train_env=env_cfg,
                eval={
                    "team": float_evals["team"],
                    "individual": float_evals["individual"][agent_id],
                    },
                actor=agent_name,
                experience=agent_name,
                partner_agents=partner_agents,
                path_to_model=critic_model_pathname,
                seed=seed,
                )
            critic_dict = OmegaConf.create(critic_dict)
            with open(critic_config_pathname, "w") as f:
                f.write(OmegaConf.to_yaml(critic_dict))

            # experience
            experience_pathname = osp.normpath(
                osp.join(zoo_path, "outputs", "experience", f"{agent_name}.pt")
                )
            config_pathname = osp.normpath(
                osp.join(zoo_path, "configs", "experience", f"{agent_name}.yaml")
                )
            os.makedirs(osp.dirname(experience_pathname), exist_ok=True)
            os.makedirs(osp.dirname(config_pathname), exist_ok=True)
            if len(self.exp_buffer) > 0:
                save_buffer = torch.concat(self.exp_buffer[agent_id])
            else:
                save_buffer = []
            torch.save(save_buffer, experience_pathname)
            buffer_dict = dict(
                max_size=self.exp_buffer_size,
                replacement_prob=self.exp_buffer_replacement_prob,
                size=len(self.exp_buffer),
                samples=len(save_buffer),
                    )
            experience_dict = dict(
                name=agent_name,
                train_env=env_cfg,
                eval=float_evals["team"],
                experience=buffer_dict,
                critics=agent_save_names,
                actors=agent_save_names,
                path_to_experience=experience_pathname,
                seed=seed,
                )
            experience_dict = OmegaConf.create(experience_dict)
            with open(config_pathname, "w") as f:
                f.write(OmegaConf.to_yaml(experience_dict))
        return agent_save_name_mapping

    @classmethod
    def from_networks(
            cls,
            actors={},
            critics={},
            action_spaces={},
            obs_spaces={},
            joint_space=None,
            actor_learning_rates={},
            critic_learning_rates={},
            gamma=1.0,
            ent_coef=0.0,
            n_steps=5,
            exp_buffer_size=1e4,
            exp_buffer_replacement_prob=1e-2,
            save_to_zoo=False,
            freeze_critic=0,
            freeze_actor=0,
            n_train_envs=1,
            ignore_trunc=True,
        ):
        ids = list(actors.keys())
        agent = cls(
            ids=ids,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_hidden_sizes={agent_id: [1] for agent_id in ids},
            critic_hidden_sizes={agent_id: [1] for agent_id in ids},
            actor_learning_rates=actor_learning_rates,
            critic_learning_rates=critic_learning_rates,
            gamma=gamma,
            ent_coef=ent_coef,
            n_steps=n_steps,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=save_to_zoo,
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=ignore_trunc,
            )

        # update the networks
        params = []
        agent.actor_nets = actors
        agent.critic_nets = critics
        for agent_id in ids:
            agent.actor_nets[agent_id] = copy(actors[agent_id])
            params.append({
                "params": agent.actor_nets[agent_id].parameters(),
                "lr": actor_learning_rates[agent_id],
                })
            agent.critic_nets[agent_id] = copy(critics[agent_id])
            params.append({
                "params": agent.critic_nets[agent_id].parameters(),
                "lr": critic_learning_rates[agent_id],
                })
        agent.optim = Adam(params,
                          lr=params[0]["lr"],
                          eps=1e-5)
        agent.saveables.update({
            "critic": agent.critic_nets,
            "actor": agent.actor_nets,
            "optim": agent.optim,
            })
        return agent

    @classmethod
    def from_config(cls, cfg, env):
        agent_ids = [a["agent_id"] for a in cfg.agents]
        assert len(agent_ids) == cfg.n_agents, \
            f"len(agent_ids) = {len(agent_ids)} (expected {cfg.n_agents})"
        action_spaces = {
            agent_id: env.action_space
            for agent_id in agent_ids
            }
        obs_spaces = {
            agent_id: env.observation_space
            for agent_id in agent_ids
            }
        if hasattr(env, "joint_space"):
            joint_space = env.joint_space
        else:
            joint_space = env.observation_space
        actor_hidden_sizes = {
            agent_id: cfg.model.actor
            for agent_id in agent_ids
            }
        critic_hidden_sizes = {
            agent_id: cfg.model.critic
            for agent_id in agent_ids
            }
        actor_learning_rates = {
            agent_id: cfg.lr.actor
            for agent_id in agent_ids
            }
        critic_learning_rates = {
            agent_id: cfg.lr.critic
            for agent_id in agent_ids
            }
        gamma = cfg.get("gamma", 1.0)
        ent_coef = cfg.get("ent_coef", 0.0)
        n_steps = cfg.get("n_steps", 5)
        clip_grad = cfg.get("clip_grad", {})
        save_to_zoo = cfg.get("save_to_zoo", False)
        ignore_trunc = cfg.get("ignore_trunc", True)
        if "exp_buffer" in cfg.keys():
            exp_buffer_size = cfg.exp_buffer.get("size", 0)
            exp_buffer_replacement_prob = cfg.exp_buffer.get("replacement_prob", 0.0)
        else:
            exp_buffer_size = 0
            exp_buffer_replacement_prob = 1.0

        if "freeze" in cfg.keys():
            freeze_critic = cfg.freeze.get("critic", 0)
            freeze_actor = cfg.freeze.get("actor", 0)
        else:
            freeze_critic = 0
            freeze_actor = 0

        if hasattr(env, "num_vec_envs"):
            n_train_envs = env.num_vec_envs
        elif hasattr(env, "vec_envs"):
            n_train_envs = len(env.vec_envs)
        else:
            n_train_envs = 1

        return cls(
            ids=agent_ids,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_hidden_sizes=actor_hidden_sizes,
            critic_hidden_sizes=critic_hidden_sizes,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rates=critic_learning_rates,
            gamma=gamma,
            ent_coef=ent_coef,
            n_steps=n_steps,
            clip_grad=clip_grad,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=save_to_zoo,
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=ignore_trunc,
            )

    @classmethod
    def load_from_zoo(cls, cfg, name_id_mapping, env, zoo_path):
        actors = {}
        critics = {}
        agent_ids = list(name_id_mapping.values())
        exp_buffers = []
        for actor_name, agent_id in name_id_mapping.items():
            # load config
            actor_cfg_path = osp.normpath(
                osp.join(zoo_path, "configs", "actors", f"{actor_name}.yaml")
                )
            actor_cfg = OmegaConf.load(actor_cfg_path)
            actors[agent_id] = torch.load(actor_cfg.path_to_model)
            # load critic network
            critic_cfg_path = osp.normpath(
                osp.join(zoo_path, "configs", "critics", f"{actor_cfg.critic}.yaml")
                )
            critic_cfg = OmegaConf.load(critic_cfg_path)
            critics[agent_id] = torch.load(critic_cfg.path_to_model)
            # load experience
            exp_path = osp.normpath(
                osp.join(zoo_path, "configs", "experience", f"{actor_cfg.experience}.yaml")
                )
            exp_cfg = OmegaConf.load(exp_path)
            exp_buffers.append(torch.load(exp_cfg.path_to_experience))

        # Load env
        action_spaces = {
            agent_id: env.action_space
            for agent_id in agent_ids
            }
        obs_spaces = {
            agent_id: env.observation_space
            for agent_id in agent_ids
            }
        if hasattr(env, "joint_space"):
            joint_space = env.joint_space
        else:
            joint_space = env.observation_space

        actor_learning_rates = {
            agent_id: cfg.lr.actor
            for agent_id in agent_ids
            }
        critic_learning_rates = {
            agent_id: cfg.lr.critic
            for agent_id in agent_ids
            }
        if "exp_buffer" in cfg.keys():
            exp_buffer_size = cfg.exp_buffer.get("size", 0)
            exp_buffer_replacement_prob = cfg.exp_buffer.get("replacement_prob", 0.0)
        else:
            exp_buffer_size = 0
            exp_buffer_replacement_prob = 1.0
        if "freeze" in cfg.keys():
            freeze_critic = cfg.freeze.get("critic", 0)
            freeze_actor = cfg.freeze.get("actor", 0)
        else:
            freeze_critic = 0
            freeze_actor = 0

        if hasattr(env, "num_vec_envs"):
            n_train_envs = env.num_vec_envs
        elif hasattr(env, "vec_envs"):
            n_train_envs = len(env.vec_envs)
        else:
            n_train_envs = 1

        agents = IA2C.from_networks(
            actors=actors,
            critics=critics,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rates=critic_learning_rates,
            gamma=cfg.gamma,
            ent_coef=cfg.ent_coef,
            n_steps=cfg.n_steps,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=cfg.get("save_to_zoo", False),
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=cfg.get("ignore_trunc", True)
            )
        return agents


class MAA2C(MultiAgent):
    def __init__(
            self,
            ids=[],
            action_spaces={},
            obs_spaces={},
            joint_space=None,
            actor_hidden_sizes={},
            critic_hidden_size=None,
            actor_learning_rates={},
            critic_learning_rate=None,
            gamma=1.0,
            ent_coef=0.0,
            clip_grad={},
            n_steps=5,
            exp_buffer_size=0,
            exp_buffer_replacement_prob=1.0,
            save_to_zoo=False,
            freeze_critic=0,
            freeze_actor=0,
            n_train_envs=1,
            ignore_trunc=True,
        ):
        super().__init__(
            ids,
            action_spaces,
            obs_spaces,
            joint_space
            )
        #state_size = obs_space.shape[1]  # When the parallel envs are in the Space
        state_size = list(obs_spaces.values())[0].shape[0]
        action_size = list(action_spaces.values())[0].n
        self.action_size = action_size

        params = []

        # Initialise the decentralised actors
        self.actor_nets = {}
        self.actor_learning_rates = {}
        for agent_id in self.ids:
            self.actor_nets[agent_id] = fc_network([
                state_size,
                *actor_hidden_sizes[agent_id],
                action_size
                ])
            self.actor_learning_rates[agent_id] = actor_learning_rates[agent_id]
            params.append({
                "params": self.actor_nets[agent_id].parameters(),
                "lr": self.actor_learning_rates[agent_id],
                })

        # Initialise the multi-agent critic
        self.critic_net = fc_network([
            self.joint_space.shape[0],
            *critic_hidden_size,
            1
            ])
        self.critic_learning_rate = critic_learning_rate
        params.append({
            "params": self.critic_net.parameters(),
            "lr": self.critic_learning_rate,
            })

        self.optim = Adam(params,
                          lr=params[0]["lr"],
                          eps=1e-5)

        # Initialise hyperparameters
        self.gamma = gamma
        self.gae_lambda = 1
        self.ent_coef = ent_coef
        self.n_steps = n_steps
        self.ignore_trunc = ignore_trunc
        self.clip_grad = defaultdict(lambda: False) 
        self.clip_grad.update(clip_grad)
        self.n_train_envs = n_train_envs
        self.freeze_critic = freeze_critic/self.n_train_envs
        self.freeze_actor = freeze_actor/self.n_train_envs
        # divide by n_train_envs since each call to update has n_train_envs transitions

        # Initialise update counters
        self.c_update = 0
        self.p_update = self.n_steps
        self.total_critic_updates = 0
        self.total_actor_updates = 0

        # Initialise memory
        self.last_transition = None
        self.last_n_transitions = ReplayBuffer(n_steps)
        self.exp_buffer_size = exp_buffer_size
        self.exp_buffer_replacement_prob = exp_buffer_replacement_prob
        self.exp_buffer = []

        self.saveables.update({
            "critic": self.critic_net,
            "actor": self.actor_nets,
            "optim": self.optim,
            })
        self.save_to_zoo = save_to_zoo

    def policy(self, obs, explore=True, agent_id=None):
        with torch.no_grad():
            if agent_id is None:
                p = {agent_id: self.policy(obs, explore=explore, agent_id=agent_id)
                     for agent_id in self.ids}
            else:
                logits = self.actor_nets[agent_id](torch.Tensor(obs[agent_id]))
                p = torch.distributions.categorical.Categorical(logits=logits)
        return p

    def act(self, obs, explore=True, agent_id=None):
        if agent_id is None:
            return {agent_id: self.act(obs, explore=explore, agent_id=agent_id)
                    for agent_id in self.ids}
        else:
            policy = self.policy(obs, explore=explore, agent_id=agent_id)
            return policy.sample().unsqueeze(1).numpy()

    def update(self):
        self.c_update += 1
        critic_frozen = self.c_update <= self.freeze_critic
        actor_frozen = self.c_update <= self.freeze_actor
        if (critic_frozen and actor_frozen) or self.c_update % self.p_update != 0:
            return {}
        # generate G, V, lp_chosen, entropy
        joint_obs, agent_obs, actions = self._stack_transitions()
        A, G, V = self._compute_gae(joint_obs)
        lp_chosens, entropies = self._compute_policy_qty(agent_obs, actions)
        # update
        self.optim.zero_grad()
        critic_loss, critic_metrics = self._update_critic(G, V, frozen=critic_frozen)
        actor_loss, actor_metrics = self._update_actor(A, lp_chosens, entropies, frozen=actor_frozen)
        total_loss = critic_loss + actor_loss
        total_loss.backward()
        if not critic_frozen:
            self.total_critic_updates += 1
            for i, p in enumerate(self.critic_net.parameters()):
                critic_metrics[f"critic_grad_{i}"] = p.grad.detach().norm().item()
        if not actor_frozen:
            self.total_actor_updates += 1
            for agent_id in self.ids:
                for i, p in enumerate(self.actor_nets[agent_id].parameters()):
                    actor_metrics[agent_id][f"actor_grad_{i}"] = p.grad.detach().norm().item()
        self.optim.step()
        # return
        train_metrics = {
            "central_critic": {
                "nTD": A.mean().item(),
                **critic_metrics
                },
            # "imp_weights": imp_weights.mean().item(),
            **actor_metrics
            }
        return train_metrics

    def _compute_gae(self, obs):
        V = self.critic_net(obs)
        with torch.no_grad():
            G = torch.zeros_like(V)
            A = torch.zeros_like(V)
            V_n = self.critic_net(self.last_transition.n_obs)
            G_n = V_n.clone()
            A_n = 0
            for i, t in enumerate(reversed(self.last_n_transitions)):
                term = t.terminated.unsqueeze(1)
                trunc = t.truncated.unsqueeze(1)
                r = t.reward.unsqueeze(1)
                if self.ignore_trunc:
                    V_n = ~(term|trunc) * V_n
                    G_n = ~(term|trunc) * G_n
                else:
                    V_n = (~trunc * (V_n * ~term)
                           + trunc * (V_n))
                    G_n = (~trunc * (G_n * ~term)
                           + trunc * (G_n))
                d = r + self.gamma * V_n - V[-i-1, :, :]
                A[-i-1, :, :] = d + self.gamma * self.gae_lambda * A_n
                G[-i-1, :, :] = r + self.gamma * G_n
                A_n = A[-i-1, :, :]
                G_n = G[-i-1, :, :]
                V_n = V[-i-1, :, :]
        return A, G, V

    def _compute_policy_qty(self, obs, acts, agent_id=None):
        if agent_id is None:
            lp_chosens, entropies = zip(*(self._compute_policy_qty(obs, acts, agent_id)
                                          for agent_id in self.ids))
            lp_chosens = dict(zip(self.ids, lp_chosens))
            entropies = dict(zip(self.ids, entropies))
            return lp_chosens, entropies
        else:
            logits = self.actor_nets[agent_id](obs[agent_id])
            lp = torch.nn.functional.log_softmax(logits, dim=2)
            lp_chosen = lp.gather(2, acts[agent_id])
            entropy = -lp.exp().mul(lp).sum(dim=2).unsqueeze(2)
            return lp_chosen, entropy

    def _update_critic(self, G, V, frozen=False):
        if frozen:
            with torch.no_grad():
                critic_loss = torch.nn.functional.mse_loss(G, V)
        else:
            critic_loss = torch.nn.functional.mse_loss(G, V)
            if self.clip_grad.get("critic"):
                torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.clip_grad["critic"])

        critic_metrics = {
            "critic_loss": critic_loss.item()
            }
        return critic_loss, critic_metrics

    def _update_actor(self, A, lp_chosen, entropy,
                      weights=1.0, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A,
                                                                         lp_chosen,
                                                                         entropy,
                                                                         weights,
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            if frozen:
                with torch.no_grad():
                    actor_loss = - (lp_chosen[agent_id].mul(A)
                                    + self.ent_coef*entropy[agent_id]).mul(weights).mean()
            else:
                actor_loss = - (lp_chosen[agent_id].mul(A)
                                + self.ent_coef*entropy[agent_id]).mul(weights).mean()
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_nets[agent_id].parameters(), self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy[agent_id].mean().item(),
                }
            return actor_loss, actor_metrics

    def store_transition(self, transitions, agent_id=None):
        # I expect this to come in as a dict of individual agent Transitions
        obs = {agent_id: torch.FloatTensor(t.obs)
               for agent_id, t in transitions.items()}
        action = {agent_id: torch.LongTensor(t.action)
                  for agent_id, t in transitions.items()}
        # Assumes common-payoff
        reward = torch.FloatTensor(list(transitions.values())[0].reward)
        terminated = torch.BoolTensor(list(transitions.values())[0].terminated)
        truncated = torch.BoolTensor(list(transitions.values())[0].truncated)
        joint_obs = torch.FloatTensor(list(transitions.values())[0].obs)
        n_obs = torch.FloatTensor(list(transitions.values())[0].n_obs)
        transition = Transition(
            obs=obs,
            action=action,
            n_obs=n_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            joint_obs=joint_obs,
            )
        self.last_transition = transition
        self.last_n_transitions.push(transition)
        self.store_exp(joint_obs)

    def _stack_transitions(self):
        joint_obs = torch.stack(list(t.joint_obs for t in self.last_n_transitions))
        agent_obs = {agent_id: torch.stack(list(t.obs[agent_id] for t in self.last_n_transitions))
                     for agent_id in self.ids}
        actions = {agent_id: torch.stack(list(t.action[agent_id] for t in self.last_n_transitions))
                   for agent_id in self.ids}
        return joint_obs, agent_obs, actions

    def store_exp(self, obs):
        if self.exp_buffer_size == 0:
            pass
        elif len(self.exp_buffer) >= self.exp_buffer_size:
            if np.random.random() < self.exp_buffer_replacement_prob:
                idx = np.random.randint(self.exp_buffer_size)
                self.exp_buffer[idx] = obs
        else:
            self.exp_buffer.append(obs)

    def add_agent(
            self,
            agent_id,
            action_space: gym.Space,
            obs_space: gym.Space,
            actor_hidden_size,
            critic_hidden_size,
            actor_learning_rate,
            critic_learning_rate,
            gamma,
            ent_coef,
            n_steps,
            save_to_zoo=False
            ):
        ...

    @property
    def info(self):
        return dict(
            type="MAA2C",
            train_steps=self.c_update*self.n_train_envs,
            actor_train_updates=self.total_actor_updates,
            critic_train_updates=self.total_critic_updates,
            actor_learning_rates=self.actor_learning_rates,
            critic_learning_rate=self.critic_learning_rate,
            gamma=self.gamma,
            ent_coef=self.ent_coef,
            clip_grad={"actor": self.clip_grad["actor"], "critic": self.clip_grad["critic"]},
            n_steps=self.n_steps,
            exp_buffer_size=self.exp_buffer_size,
            exp_buffer_replacement_prob=self.exp_buffer_replacement_prob,
            ignore_trunc=self.ignore_trunc
            )

    def zoo_save(self, zoo_path, agent_save_names, env_cfg, evals, seed):
        agent_save_name_mapping = dict(zip(self.ids, agent_save_names))
        concat_name = "_".join(agent_save_names)
        # configs
        # actors
        # inelegant way of changing the evals into floats (from numpy.floats)
        # required so omegaconf can read
        float_evals = {
                "team": {k: v.item() for k, v in evals["team"].items()},
                "individual": {k: {k: v.item() for k, v in v.items()}
                               for k, v in evals["individual"].items()},
                }
        info = self.info
        for agent_id in self.ids:
            agent_name = agent_save_name_mapping[agent_id]
            model_pathname = osp.normpath(
                osp.join(zoo_path, "outputs", "actors", f"{agent_name}.pt")
                )
            config_pathname = osp.normpath(
                osp.join(zoo_path, "configs", "actors", f"{agent_name}.yaml")
                )
            os.makedirs(osp.dirname(model_pathname), exist_ok=True)
            os.makedirs(osp.dirname(config_pathname), exist_ok=True)
            torch.save(self.actor_nets[agent_id], model_pathname)
            partner_agents = [agent_save_name_mapping[other_agent_id]
                              for other_agent_id in self.ids
                              if other_agent_id != agent_id]
            model_dict = dict(
                internal_name=agent_id,
                train_steps=info["train_steps"],
                train_updates=info["actor_train_updates"],
                learning_rate=info["actor_learning_rates"][agent_id],
                gamma=self.gamma,
                ent_coef=self.ent_coef,
                clip_grad=self.clip_grad["actor"],
                n_steps=self.n_steps,
                ignore_trunc=self.ignore_trunc,
                )
            actor_dict = dict(
                name=agent_name,
                model=model_dict,
                train_env=env_cfg,
                eval={
                    "team": float_evals["team"],
                    "individual": float_evals["individual"][agent_id],
                    },
                critic=concat_name,
                experience=concat_name,
                partner_agents=partner_agents,
                path_to_model=model_pathname,
                seed=seed,
                )
            actor_dict = OmegaConf.create(actor_dict)
            with open(config_pathname, "w") as f:
                f.write(OmegaConf.to_yaml(actor_dict))

        # critic
        model_pathname = osp.normpath(
            osp.join(zoo_path, "outputs", "critics", f"{concat_name}.pt")
            )
        config_pathname = osp.normpath(
            osp.join(zoo_path, "configs", "critics", f"{concat_name}.yaml")
            )
        os.makedirs(osp.dirname(model_pathname), exist_ok=True)
        os.makedirs(osp.dirname(config_pathname), exist_ok=True)
        torch.save(self.critic_net, model_pathname)
        model_dict = dict(
            train_steps=info["train_steps"],
            train_updates=info["critic_train_updates"],
            learning_rate=info["critic_learning_rate"],
            gamma=self.gamma,
            clip_grad=self.clip_grad["critic"],
            n_steps=self.n_steps,
            ignore_trunc=self.ignore_trunc,
            )
        critic_dict = dict(
            name=concat_name,
            model=model_dict,
            train_env=env_cfg,
            eval=float_evals["team"],
            actors=agent_save_names,
            experience=concat_name,
            path_to_model=model_pathname,
            seed=seed,
            )
        critic_dict = OmegaConf.create(critic_dict)
        with open(config_pathname, "w") as f:
            f.write(OmegaConf.to_yaml(critic_dict))
        # experience
        experience_pathname = osp.normpath(
            osp.join(zoo_path, "outputs", "experience", f"{concat_name}.pt")
            )
        config_pathname = osp.normpath(
            osp.join(zoo_path, "configs", "experience", f"{concat_name}.yaml")
            )
        os.makedirs(osp.dirname(experience_pathname), exist_ok=True)
        os.makedirs(osp.dirname(config_pathname), exist_ok=True)
        if len(self.exp_buffer) > 0:
            save_buffer = torch.concat(self.exp_buffer)
        else:
            save_buffer = []
        torch.save(save_buffer, experience_pathname)
        buffer_dict = dict(
            max_size=self.exp_buffer_size,
            replacement_prob=self.exp_buffer_replacement_prob,
            size=len(self.exp_buffer),
            samples=len(save_buffer),
                )
        experience_dict = dict(
            name=concat_name,
            train_env=env_cfg,
            eval=float_evals["team"],
            experience=buffer_dict,
            critic=concat_name,
            actors=agent_save_names,
            path_to_experience=experience_pathname,
            seed=seed,
            )
        experience_dict = OmegaConf.create(experience_dict)
        with open(config_pathname, "w") as f:
            f.write(OmegaConf.to_yaml(experience_dict))
        return agent_save_name_mapping

    @classmethod
    def from_networks(
            cls,
            networks={},
            action_spaces={},
            obs_spaces={},
            joint_space=None,
            actor_learning_rates={},
            critic_learning_rate=None,
            gamma=1.0,
            ent_coef=0.0,
            n_steps=5,
            exp_buffer_size=1e4,
            exp_buffer_replacement_prob=1e-2,
            save_to_zoo=False,
            freeze_critic=0,
            freeze_actor=0,
            n_train_envs=1,
            ignore_trunc=True,
        ):
        ids = [agent_id for agent_id in networks.keys() if agent_id != "central_critic"]
        agent = cls(
            ids=ids,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_hidden_sizes={agent_id: [1] for agent_id in ids},
            critic_hidden_size=[1],
            actor_learning_rates=actor_learning_rates,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            ent_coef=ent_coef,
            n_steps=n_steps,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=save_to_zoo,
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=ignore_trunc,
            )

        # update the networks
        params = []
        for agent_id in ids:
            agent.actor_nets[agent_id] = copy(networks[agent_id])
            params.append({
                "params": agent.actor_nets[agent_id].parameters(),
                "lr": actor_learning_rates[agent_id],
                })
        agent.critic_net = copy(networks["central_critic"])
        params.append({
            "params": agent.critic_net.parameters(),
            "lr": critic_learning_rate,
            })
        agent.optim = Adam(params,
                          lr=params[0]["lr"],
                          eps=1e-5)
        agent.saveables.update({
            "critic": agent.critic_net,
            "actor": agent.actor_nets,
            "optim": agent.optim,
            })
        return agent

    @classmethod
    def from_config(cls, cfg, env):
        agent_ids = [a["agent_id"] for a in cfg.agents]
        assert len(agent_ids) == cfg.n_agents, \
            f"len(agent_ids) = {len(agent_ids)} (expected {cfg.n_agents})"
        action_spaces = {
            agent_id: env.action_space
            for agent_id in agent_ids
            }
        obs_spaces = {
            agent_id: env.observation_space
            for agent_id in agent_ids
            }
        if hasattr(env, "joint_space"):
            joint_space = env.joint_space
        else:
            joint_space = env.observation_space
        actor_hidden_sizes = {
            agent_id: cfg.model.actor
            for agent_id in agent_ids
            }
        critic_hidden_size = cfg.model.critic
        actor_learning_rates = {
            agent_id: cfg.lr.actor
            for agent_id in agent_ids
            }
        critic_learning_rate = cfg.lr.critic
        gamma = cfg.get("gamma", 1.0)
        ent_coef = cfg.get("ent_coef", 0.0)
        n_steps = cfg.get("n_steps", 5)
        clip_grad = cfg.get("clip_grad", {})
        save_to_zoo = cfg.get("save_to_zoo", False)
        ignore_trunc = cfg.get("ignore_trunc", True)
        if "exp_buffer" in cfg.keys():
            exp_buffer_size = cfg.exp_buffer.get("size", 0)
            exp_buffer_replacement_prob = cfg.exp_buffer.get("replacement_prob", 0.0)
        else:
            exp_buffer_size = 0
            exp_buffer_replacement_prob = 1.0

        if "freeze" in cfg.keys():
            freeze_critic = cfg.freeze.get("critic", 0)
            freeze_actor = cfg.freeze.get("actor", 0)
        else:
            freeze_critic = 0
            freeze_actor = 0

        if hasattr(env, "num_vec_envs"):
            n_train_envs = env.num_vec_envs
        elif hasattr(env, "vec_envs"):
            n_train_envs = len(env.vec_envs)
        else:
            n_train_envs = 1

        return cls(
            ids=agent_ids,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_hidden_sizes=actor_hidden_sizes,
            critic_hidden_size=critic_hidden_size,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            ent_coef=ent_coef,
            n_steps=n_steps,
            clip_grad=clip_grad,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=save_to_zoo,
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=ignore_trunc,
            )

    @classmethod
    def from_zoo(cls, actor_names, env, zoo_path, override_cfg={}, save_to_zoo=False):
        networks = {}
        critic_names = set()
        name_id_mapping = {}
        agent_ids = []
        actor_learning_rates = {}

        for actor_name in actor_names:
            # load config
            path = osp.normpath(
                osp.join(zoo_path, "configs", "actors", f"{actor_name}.yaml")
                )
            cfg = OmegaConf.load(path)
            agent_id = cfg.model.internal_name
            agent_ids.append(agent_id)
            name_id_mapping[actor_name] = agent_id
            critic_names.add(cfg.critic)
            actor_learning_rates[agent_id] = cfg.model.learning_rate
            ent_coef = cfg.model.ent_coef
            networks[agent_id] = torch.load(cfg.path_to_model)


        if len(cfg.partner_agents) + 1 < len(actor_names):
            raise Exception("MAA2C.from_zoo method got fewer actors than expected")

        if len(critic_names) != 1:
            raise Exception(
                f"""MAA2C.from_zoo method only applies to agents with a shared critic.
                Got critics {critic_names}.
                """
                )

        # Load env
        action_spaces = {
            agent_id: env.action_space
            for agent_id in agent_ids
            }
        obs_spaces = {
            agent_id: env.observation_space
            for agent_id in agent_ids
            }

        if hasattr(env, "joint_space"):
            joint_space = env.joint_space
        else:
            joint_space = env.observation_space

        # Load critic
        critic_name = critic_names.pop()
        path = osp.normpath(
            osp.join(zoo_path, "configs", "critics", f"{critic_name}.yaml")
            )
        cfg = OmegaConf.load(path)
        critic_learning_rate = cfg.model.learning_rate
        networks["central_critic"] = torch.load(cfg.path_to_model)
        ignore_trunc = cfg.model.get("ignore_trunc", True)

        actor_learning_rates = override_cfg.get("actor_learning_rates", actor_learning_rates)
        critic_learning_rate = override_cfg.get("critic_learning_rate", critic_learning_rate)
        gamma = override_cfg.get("gamma", cfg.model.gamma)
        ent_coef = override_cfg.get("ent_coef", ent_coef)
        n_steps = override_cfg.get("n_steps", cfg.model.n_steps)
        ignore_trunc = override_cfg.get("ignore_trunc", ignore_trunc)
        # Doesn't make sense to load these from the zoo
        freeze_critic = override_cfg.get("freeze_critic", 0)
        freeze_actor = override_cfg.get("freeze_actor", 0)

        # Load experience config
        path = osp.normpath(
            osp.join(zoo_path, "configs", "experience", f"{cfg.experience}.yaml")
            )
        cfg = OmegaConf.load(path)
        exp_buffer_size = override_cfg.get("exp_buffer_size", cfg.experience.max_size)
        exp_buffer_replacement_prob = \
            override_cfg.get("exp_buffer_replacement_prob", cfg.experience.replacement_prob)

        if hasattr(env, "num_vec_envs"):
            n_train_envs = env.num_vec_envs
        elif hasattr(env, "vec_envs"):
            n_train_envs = len(env.vec_envs)
        else:
            n_train_envs = 1

        agents = MAA2C.from_networks(
            networks=networks,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            ent_coef=ent_coef,
            n_steps=n_steps,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=save_to_zoo,
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=ignore_trunc,
            )
        return agents



class IPPO(MultiAgent):
    def __init__(
            self,
            ids=[],
            action_spaces={},
            obs_spaces={},
            joint_space=None,
            actor_hidden_sizes={},
            critic_hidden_sizes={},
            actor_learning_rates={},
            critic_learning_rates={},
            gamma=1.0,
            gae_lambda=1.0,
            n_epochs=1,
            clip_coef=0.1,
            ent_coef=0.0,
            clip_grad={},
            n_steps=5,
            exp_buffer_size=0,
            exp_buffer_replacement_prob=1.0,
            save_to_zoo=False,
            freeze_critic=0,
            freeze_actor=0,
            n_train_envs=1,
            ignore_trunc=True,
        ):
        super().__init__(
            ids,
            action_spaces,
            obs_spaces,
            joint_space
            )
        state_size = list(obs_spaces.values())[0].shape[0]
        action_size = list(action_spaces.values())[0].n
        self.action_size = action_size

        params = []

        # Initialise the decentralised actors & critics
        self.actor_nets = {}
        self.actor_learning_rates = {}
        self.critic_nets = {}
        self.critic_learning_rates = {}
        for agent_id in self.ids:
            # Actors
            self.actor_nets[agent_id] = fc_network([
                state_size,
                *actor_hidden_sizes[agent_id],
                action_size
                ])
            self.actor_learning_rates[agent_id] = actor_learning_rates[agent_id]
            params.append({
                "params": self.actor_nets[agent_id].parameters(),
                "lr": self.actor_learning_rates[agent_id],
                })
            # Critics
            self.critic_nets[agent_id] = fc_network([
                state_size,
                *critic_hidden_sizes[agent_id],
                1
                ])
            self.critic_learning_rates[agent_id] = critic_learning_rates[agent_id]
            params.append({
                "params": self.critic_nets[agent_id].parameters(),
                "lr": self.critic_learning_rates[agent_id],
                })

        self.optim = Adam(params,
                          lr=params[0]["lr"],
                          eps=1e-5)

        # Initialise hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.n_steps = n_steps
        self.ignore_trunc = ignore_trunc
        self.clip_grad = defaultdict(lambda: False) 
        self.clip_grad.update(clip_grad)
        self.n_train_envs = n_train_envs
        self.freeze_critic = freeze_critic/self.n_train_envs
        self.freeze_actor = freeze_actor/self.n_train_envs
        # divide by n_train_envs since each call to update has n_train_envs transitions

        # Initialise update counters
        self.c_update = 0
        self.p_update = self.n_steps
        self.total_critic_updates = 0
        self.total_actor_updates = 0

        # Initialise memory
        self.last_transition = None
        self.last_n_transitions = ReplayBuffer(n_steps)
        self.exp_buffer_size = exp_buffer_size
        self.exp_buffer_replacement_prob = exp_buffer_replacement_prob
        self.exp_buffer = {agent_id: [] for agent_id in self.ids}

        self.saveables.update({
            "critic": self.critic_nets,
            "actor": self.actor_nets,
            "optim": self.optim,
            })
        self.save_to_zoo = save_to_zoo

    def policy(self, obs, explore=True, agent_id=None):
        with torch.no_grad():
            if agent_id is None:
                p = {agent_id: self.policy(obs[agent_id], explore=explore, agent_id=agent_id)
                     for agent_id in self.ids}
            else:
                logits = self.actor_nets[agent_id](torch.Tensor(obs[agent_id]))
                p = torch.distributions.categorical.Categorical(logits=logits)
        return p

    def act(self, obs, explore=True, agent_id=None):
        if agent_id is None:
            return {agent_id: self.act(obs, explore=explore, agent_id=agent_id)
                    for agent_id in self.ids}
        else:
            policy = self.policy(obs, explore=explore, agent_id=agent_id)
            return policy.sample().unsqueeze(1).numpy()

    def update(self):
        self.c_update += 1
        critic_frozen = self.c_update <= self.freeze_critic
        actor_frozen = self.c_update <= self.freeze_actor
        if (critic_frozen and actor_frozen) or self.c_update % self.p_update != 0:
            return {}
        # generate G, V, lp_chosen, entropy
        _, agent_obs, actions = self._stack_transitions()
        As, Gs, Vs = self._compute_gae(agent_obs)
        with torch.no_grad():
            orig_lp_chosens, _ = self._compute_policy_qty(agent_obs, actions)
        # update loop:
        for epoch in range(self.n_epochs):
            self.optim.zero_grad()
            new_lp_chosens, new_entropies = self._compute_policy_qty(agent_obs, actions)
            policy_ratio = {agent_id: torch.exp(new_lp_chosens[agent_id] - orig_lp_chosens[agent_id])
                            for agent_id in self.ids}
            new_Vs = {agent_id: self.critic_nets[agent_id](agent_obs[agent_id])
                      for agent_id in self.ids}
            critic_loss, critic_metrics = self._update_critic(Gs, new_Vs, frozen=critic_frozen)
            actor_loss, actor_metrics = self._update_actor(As, policy_ratio, new_entropies, frozen=actor_frozen)
            total_loss = critic_loss + actor_loss
            total_loss.backward()
            self.optim.step()
            if not critic_frozen:
                self.total_critic_updates += 1
                for agent_id in self.ids:
                    for i, p in enumerate(self.critic_nets[agent_id].parameters()):
                        critic_metrics[agent_id][f"critic_grad_{i}"] = p.grad.detach().norm().item()
            if not actor_frozen:
                self.total_actor_updates += 1
                for agent_id in self.ids:
                    for i, p in enumerate(self.actor_nets[agent_id].parameters()):
                        actor_metrics[agent_id][f"actor_grad_{i}"] = p.grad.detach().norm().item()

        # return
        train_metrics = {
            **critic_metrics,
            **actor_metrics
            }
        return train_metrics

    def _compute_gae(self, obs, agent_id=None):
        if agent_id is None:
            As, Gs, Vs = zip(*(self._compute_gae(obs, agent_id=agent_id)
                           for agent_id in self.ids))
            As = dict(zip(self.ids, As))
            Gs = dict(zip(self.ids, Gs))
            Vs = dict(zip(self.ids, Vs))
            return As, Gs, Vs
        else:
            V = self.critic_nets[agent_id](obs[agent_id])
            with torch.no_grad():
                G = torch.zeros_like(V)
                A = torch.zeros_like(V)
                V_n = self.critic_nets[agent_id](self.last_transition.n_obs[agent_id])
                G_n = V_n.clone()
                A_n = 0
                for i, t in enumerate(reversed(self.last_n_transitions)):
                    term = t.terminated.unsqueeze(1)
                    trunc = t.truncated.unsqueeze(1)
                    r = t.reward.unsqueeze(1)
                    if self.ignore_trunc:
                        V_n = ~(term|trunc) * V_n
                        G_n = ~(term|trunc) * G_n
                    else:
                        V_n = (~trunc * (V_n * ~term)
                               + trunc * (V_n))
                        G_n = (~trunc * (G_n * ~term)
                               + trunc * (G_n))
                    d = r + self.gamma * V_n - V[-i-1, :, :]
                    A[-i-1, :, :] = d + self.gamma * self.gae_lambda * A_n
                    G[-i-1, :, :] = r + self.gamma * G_n
                    A_n = A[-i-1, :, :]
                    G_n = G[-i-1, :, :]
                    V_n = V[-i-1, :, :]
            return A, G, V

    def _compute_policy_qty(self, obs, acts, agent_id=None):
        if agent_id is None:
            lp_chosens, entropies = zip(*(self._compute_policy_qty(obs, acts, agent_id)
                                          for agent_id in self.ids))
            lp_chosens = dict(zip(self.ids, lp_chosens))
            entropies = dict(zip(self.ids, entropies))
            return lp_chosens, entropies
        else:
            logits = self.actor_nets[agent_id](obs[agent_id])
            lp = torch.nn.functional.log_softmax(logits, dim=2)
            lp_chosen = lp.gather(2, acts[agent_id])
            entropy = -lp.exp().mul(lp).sum(dim=2).unsqueeze(2)
            return lp_chosen, entropy

    def _update_critic(self, G, V, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            critic_metrics = {}
            for agent_id in self.ids:
                critic_loss, critic_metrics[agent_id] = self._update_critic(G[agent_id], V[agent_id],
                                                                            agent_id=agent_id,
                                                                            frozen=frozen)
                total_loss += critic_loss
            return total_loss, critic_metrics
        else:
            if frozen:
                with torch.no_grad():
                    critic_loss = torch.nn.functional.mse_loss(G, V)
            else:
                critic_loss = torch.nn.functional.mse_loss(G, V)
                if self.clip_grad.get("critic"):
                    torch.nn.utils.clip_grad_norm_(self.critic_nets[agent_id].parameters(), self.clip_grad["critic"])
            critic_metrics = {
                "critic_loss": critic_loss.item()
                }
            return critic_loss, critic_metrics

    def _update_actor(self, A, ratio, entropy, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A[agent_id],
                                                                         ratio[agent_id],
                                                                         entropy[agent_id],
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            A_norm = (A-A.mean())/(A.std() + 1e-8)
            if frozen:
                with torch.no_grad():
                    policy_loss = torch.min(
                        A_norm * ratio,
                        A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                        ).mean()
                    actor_loss = -(policy_loss + self.ent_coef*entropy.mean())
            else:
                policy_loss = torch.min(
                    A_norm * ratio,
                    A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                    ).mean()
                actor_loss = -(policy_loss + self.ent_coef*entropy.mean())
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_nets[agent_id].parameters(), self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy.mean().item(),
                }
            return actor_loss, actor_metrics

    def store_transition(self, transitions, agent_id=None):
        # I expect this to come in as a dict of individual agent Transitions
        obs = {agent_id: torch.FloatTensor(t.obs)
               for agent_id, t in transitions.items()}
        n_obs = {agent_id: torch.FloatTensor(t.n_obs)
               for agent_id, t in transitions.items()}
        action = {agent_id: torch.LongTensor(t.action)
                  for agent_id, t in transitions.items()}
        # Assumes common-payoff
        reward = torch.FloatTensor(list(transitions.values())[0].reward)
        terminated = torch.BoolTensor(list(transitions.values())[0].terminated)
        truncated = torch.BoolTensor(list(transitions.values())[0].truncated)
        joint_obs = torch.FloatTensor(list(transitions.values())[0].obs)
        transition = Transition(
            obs=obs,
            action=action,
            n_obs=n_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            joint_obs=joint_obs,
            )
        self.last_transition = transition
        self.last_n_transitions.push(transition)
        self.store_exp(obs)

    def _stack_transitions(self):
        joint_obs = torch.stack(list(t.joint_obs for t in self.last_n_transitions))
        agent_obs = {agent_id: torch.stack(list(t.obs[agent_id] for t in self.last_n_transitions))
                     for agent_id in self.ids}
        actions = {agent_id: torch.stack(list(t.action[agent_id] for t in self.last_n_transitions))
                   for agent_id in self.ids}
        return joint_obs, agent_obs, actions

    def store_exp(self, obs, agent_id=None):
        if agent_id is None:
            for agent_id in self.ids:
                self.store_exp(obs, agent_id=agent_id)
        else:
            if self.exp_buffer_size == 0:
                pass
            elif len(self.exp_buffer[agent_id]) >= self.exp_buffer_size:
                if np.random.random() < self.exp_buffer_replacement_prob:
                    idx = np.random.randint(self.exp_buffer_size)
                    self.exp_buffer[idx] = obs[agent_id]
            else:
                self.exp_buffer[agent_id].append(obs[agent_id])

    def add_agent(
            self,
            agent_id,
            action_space: gym.Space,
            obs_space: gym.Space,
            actor_hidden_size,
            critic_hidden_size,
            actor_learning_rate,
            critic_learning_rate,
            gamma,
            ent_coef,
            n_steps,
            save_to_zoo=False
            ):
        ...

    @property
    def info(self):
        return dict(
            type="IPPO",
            train_steps=self.c_update*self.n_train_envs,
            actor_train_updates=self.total_actor_updates,
            critic_train_updates=self.total_critic_updates,
            actor_learning_rates=self.actor_learning_rates,
            critic_learning_rates=self.critic_learning_rates,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_epochs=self.n_epochs,
            clip_coef=self.clip_coef,
            ent_coef=self.ent_coef,
            clip_grad={"actor": self.clip_grad["actor"], "critic": self.clip_grad["critic"]},
            n_steps=self.n_steps,
            exp_buffer_size=self.exp_buffer_size,
            exp_buffer_replacement_prob=self.exp_buffer_replacement_prob,
            ignore_trunc=self.ignore_trunc
            )

    def zoo_save(self, zoo_path, agent_save_names, env_cfg, evals, seed):
        agent_save_name_mapping = dict(zip(self.ids, agent_save_names))
        concat_name = "_".join(agent_save_names)
        # configs
        # actors
        # inelegant way of changing the evals into floats (from numpy.floats)
        # required so omegaconf can read
        float_evals = {
                "team": {k: v.item() for k, v in evals["team"].items()},
                "individual": {k: {k: v.item() for k, v in v.items()}
                               for k, v in evals["individual"].items()},
                }
        info = self.info
        for agent_id in self.ids:
            agent_name = agent_save_name_mapping[agent_id]
            partner_agents = [agent_save_name_mapping[other_agent_id]
                              for other_agent_id in self.ids
                              if other_agent_id != agent_id]
            # Actor
            actor_model_pathname = osp.normpath(
                osp.join(zoo_path, "outputs", "actors", f"{agent_name}.pt")
                )
            actor_config_pathname = osp.normpath(
                osp.join(zoo_path, "configs", "actors", f"{agent_name}.yaml")
                )
            os.makedirs(osp.dirname(actor_model_pathname), exist_ok=True)
            os.makedirs(osp.dirname(actor_config_pathname), exist_ok=True)
            torch.save(self.actor_nets[agent_id], actor_model_pathname)
            actor_model_dict = dict(
                internal_name=agent_id,
                train_steps=info["train_steps"],
                train_updates=info["actor_train_updates"],
                learning_rate=info["actor_learning_rates"][agent_id],
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_epochs=self.n_epochs,
                clip_coef=self.clip_coef,
                ent_coef=self.ent_coef,
                clip_grad=self.clip_grad["actor"],
                n_steps=self.n_steps,
                ignore_trunc=self.ignore_trunc,
                )
            actor_dict = dict(
                name=agent_name,
                model=actor_model_dict,
                train_env=env_cfg,
                eval={
                    "team": float_evals["team"],
                    "individual": float_evals["individual"][agent_id],
                    },
                critic=agent_name,
                experience=agent_name,
                partner_agents=partner_agents,
                path_to_model=actor_model_pathname,
                seed=seed,
                )
            actor_dict = OmegaConf.create(actor_dict)
            with open(actor_config_pathname, "w") as f:
                f.write(OmegaConf.to_yaml(actor_dict))

            # Critic
            critic_model_pathname = osp.normpath(
                osp.join(zoo_path, "outputs", "critics", f"{agent_name}.pt")
                )
            critic_config_pathname = osp.normpath(
                osp.join(zoo_path, "configs", "critics", f"{agent_name}.yaml")
                )
            os.makedirs(osp.dirname(critic_model_pathname), exist_ok=True)
            os.makedirs(osp.dirname(critic_config_pathname), exist_ok=True)
            torch.save(self.critic_nets[agent_id], critic_model_pathname)
            critic_model_dict = dict(
                internal_name=agent_id,
                train_steps=info["train_steps"],
                train_updates=info["critic_train_updates"],
                learning_rate=info["critic_learning_rates"][agent_id],
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_epochs=self.n_epochs,
                clip_coef=self.clip_coef,
                ent_coef=self.ent_coef,
                clip_grad=self.clip_grad["critic"],
                n_steps=self.n_steps,
                ignore_trunc=self.ignore_trunc,
                )
            critic_dict = dict(
                name=agent_name,
                model=critic_model_dict,
                train_env=env_cfg,
                eval={
                    "team": float_evals["team"],
                    "individual": float_evals["individual"][agent_id],
                    },
                actor=agent_name,
                experience=agent_name,
                partner_agents=partner_agents,
                path_to_model=critic_model_pathname,
                seed=seed,
                )
            critic_dict = OmegaConf.create(critic_dict)
            with open(critic_config_pathname, "w") as f:
                f.write(OmegaConf.to_yaml(critic_dict))

            # experience
            experience_pathname = osp.normpath(
                osp.join(zoo_path, "outputs", "experience", f"{agent_name}.pt")
                )
            config_pathname = osp.normpath(
                osp.join(zoo_path, "configs", "experience", f"{agent_name}.yaml")
                )
            os.makedirs(osp.dirname(experience_pathname), exist_ok=True)
            os.makedirs(osp.dirname(config_pathname), exist_ok=True)
            if len(self.exp_buffer) > 0:
                save_buffer = torch.concat(self.exp_buffer[agent_id])
            else:
                save_buffer = []
            torch.save(save_buffer, experience_pathname)
            buffer_dict = dict(
                max_size=self.exp_buffer_size,
                replacement_prob=self.exp_buffer_replacement_prob,
                size=len(self.exp_buffer),
                samples=len(save_buffer),
                    )
            experience_dict = dict(
                name=agent_name,
                train_env=env_cfg,
                eval=float_evals["team"],
                experience=buffer_dict,
                critics=agent_save_names,
                actors=agent_save_names,
                path_to_experience=experience_pathname,
                seed=seed,
                )
            experience_dict = OmegaConf.create(experience_dict)
            with open(config_pathname, "w") as f:
                f.write(OmegaConf.to_yaml(experience_dict))
        return agent_save_name_mapping

    @classmethod
    def from_networks(
            cls,
            actors={},
            critics={},
            action_spaces={},
            obs_spaces={},
            joint_space=None,
            actor_learning_rates={},
            critic_learning_rates={},
            gamma=1.0,
            gae_lambda=1.0,
            n_epochs=1,
            clip_coef=0.1,
            ent_coef=0.0,
            n_steps=5,
            exp_buffer_size=1e4,
            exp_buffer_replacement_prob=1e-2,
            save_to_zoo=False,
            freeze_critic=0,
            freeze_actor=0,
            n_train_envs=1,
            ignore_trunc=True,
        ):
        ids = list(actors.keys())
        agent = cls(
            ids=ids,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_hidden_sizes={agent_id: [1] for agent_id in ids},
            critic_hidden_sizes={agent_id: [1] for agent_id in ids},
            actor_learning_rates=actor_learning_rates,
            critic_learning_rates=critic_learning_rates,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_epochs=n_epochs,
            clip_coef=clip_coef,
            ent_coef=ent_coef,
            n_steps=n_steps,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=save_to_zoo,
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=ignore_trunc,
            )

        # update the networks
        params = []
        agent.actor_nets = actors
        agent.critic_nets = critics
        for agent_id in ids:
            agent.actor_nets[agent_id] = copy(actors[agent_id])
            params.append({
                "params": agent.actor_nets[agent_id].parameters(),
                "lr": actor_learning_rates[agent_id],
                })
            agent.critic_nets[agent_id] = copy(critics[agent_id])
            params.append({
                "params": agent.critic_nets[agent_id].parameters(),
                "lr": critic_learning_rates[agent_id],
                })
        agent.optim = Adam(params,
                          lr=params[0]["lr"],
                          eps=1e-5)
        agent.saveables.update({
            "critic": agent.critic_nets,
            "actor": agent.actor_nets,
            "optim": agent.optim,
            })
        return agent

    @classmethod
    def from_config(cls, cfg, env):
        agent_ids = [a["agent_id"] for a in cfg.agents]
        assert len(agent_ids) == cfg.n_agents, \
            f"len(agent_ids) = {len(agent_ids)} (expected {cfg.n_agents})"
        action_spaces = {
            agent_id: env.action_space
            for agent_id in agent_ids
            }
        obs_spaces = {
            agent_id: env.observation_space
            for agent_id in agent_ids
            }
        if hasattr(env, "joint_space"):
            joint_space = env.joint_space
        else:
            joint_space = env.observation_space
        actor_hidden_sizes = {
            agent_id: cfg.model.actor
            for agent_id in agent_ids
            }
        critic_hidden_sizes = {
            agent_id: cfg.model.critic
            for agent_id in agent_ids
            }
        actor_learning_rates = {
            agent_id: cfg.lr.actor
            for agent_id in agent_ids
            }
        critic_learning_rates = {
            agent_id: cfg.lr.critic
            for agent_id in agent_ids
            }
        gamma = cfg.get("gamma", 1.0)
        gae_lambda = cfg.get("gae_lambda", 1)
        n_epochs = cfg.get("n_epochs", 1)
        clip_coef = cfg.get("clip_coef", 0.1)
        ent_coef = cfg.get("ent_coef", 0.0)
        n_steps = cfg.get("n_steps", 5)
        clip_grad = cfg.get("clip_grad", {})
        save_to_zoo = cfg.get("save_to_zoo", False)
        ignore_trunc = cfg.get("ignore_trunc", True)
        if "exp_buffer" in cfg.keys():
            exp_buffer_size = cfg.exp_buffer.get("size", 0)
            exp_buffer_replacement_prob = cfg.exp_buffer.get("replacement_prob", 0.0)
        else:
            exp_buffer_size = 0
            exp_buffer_replacement_prob = 1.0

        if "freeze" in cfg.keys():
            freeze_critic = cfg.freeze.get("critic", 0)
            freeze_actor = cfg.freeze.get("actor", 0)
        else:
            freeze_critic = 0
            freeze_actor = 0

        if hasattr(env, "num_vec_envs"):
            n_train_envs = env.num_vec_envs
        elif hasattr(env, "vec_envs"):
            n_train_envs = len(env.vec_envs)
        else:
            n_train_envs = 1

        return cls(
            ids=agent_ids,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_hidden_sizes=actor_hidden_sizes,
            critic_hidden_sizes=critic_hidden_sizes,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rates=critic_learning_rates,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_epochs=n_epochs,
            clip_coef=clip_coef,
            ent_coef=ent_coef,
            n_steps=n_steps,
            clip_grad=clip_grad,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=save_to_zoo,
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=ignore_trunc,
            )

    @classmethod
    def load_from_zoo(cls, cfg, name_id_mapping, env, zoo_path):
        actors = {}
        critics = {}
        agent_ids = list(name_id_mapping.values())
        exp_buffers = []
        for actor_name, agent_id in name_id_mapping.items():
            # load config
            actor_cfg_path = osp.normpath(
                osp.join(zoo_path, "configs", "actors", f"{actor_name}.yaml")
                )
            actor_cfg = OmegaConf.load(actor_cfg_path)
            actors[agent_id] = torch.load(actor_cfg.path_to_model)
            # load critic network
            critic_cfg_path = osp.normpath(
                osp.join(zoo_path, "configs", "critics", f"{actor_cfg.critic}.yaml")
                )
            critic_cfg = OmegaConf.load(critic_cfg_path)
            critics[agent_id] = torch.load(critic_cfg.path_to_model)
            # load experience
            exp_path = osp.normpath(
                osp.join(zoo_path, "configs", "experience", f"{actor_cfg.experience}.yaml")
                )
            exp_cfg = OmegaConf.load(exp_path)
            exp_buffers.append(torch.load(exp_cfg.path_to_experience))

        # Load env
        action_spaces = {
            agent_id: env.action_space
            for agent_id in agent_ids
            }
        obs_spaces = {
            agent_id: env.observation_space
            for agent_id in agent_ids
            }
        if hasattr(env, "joint_space"):
            joint_space = env.joint_space
        else:
            joint_space = env.observation_space

        actor_learning_rates = {
            agent_id: cfg.lr.actor
            for agent_id in agent_ids
            }
        critic_learning_rates = {
            agent_id: cfg.lr.critic
            for agent_id in agent_ids
            }
        if "exp_buffer" in cfg.keys():
            exp_buffer_size = cfg.exp_buffer.get("size", 0)
            exp_buffer_replacement_prob = cfg.exp_buffer.get("replacement_prob", 0.0)
        else:
            exp_buffer_size = 0
            exp_buffer_replacement_prob = 1.0
        if "freeze" in cfg.keys():
            freeze_critic = cfg.freeze.get("critic", 0)
            freeze_actor = cfg.freeze.get("actor", 0)
        else:
            freeze_critic = 0
            freeze_actor = 0

        if hasattr(env, "num_vec_envs"):
            n_train_envs = env.num_vec_envs
        elif hasattr(env, "vec_envs"):
            n_train_envs = len(env.vec_envs)
        else:
            n_train_envs = 1

        agents = cls.from_networks(
            actors=actors,
            critics=critics,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rates=critic_learning_rates,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            n_epochs=cfg.n_epochs,
            clip_coef=cfg.clip_coef,
            ent_coef=cfg.ent_coef,
            n_steps=cfg.n_steps,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=cfg.get("save_to_zoo", False),
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=cfg.get("ignore_trunc", True)
            )
        return agents


class DeIPPO(IPPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self):
        self.c_update += 1
        critic_frozen = self.c_update <= self.freeze_critic
        actor_frozen = self.c_update <= self.freeze_actor
        if (critic_frozen and actor_frozen) or self.c_update % self.p_update != 0:
            return {}
        # generate G, V, lp_chosen, entropy
        _, agent_obs, actions = self._stack_transitions()
        As, Gs, Vs = self._compute_gae(agent_obs)
        with torch.no_grad():
            behav_lp_chosens, _ = self._compute_policy_qty(agent_obs, actions)
            proxi_lp_chosens = behav_lp_chosens
        # update loop:
        for epoch in range(self.n_epochs):
            self.optim.zero_grad()
            new_lp_chosens, new_entropies = self._compute_policy_qty(agent_obs, actions)
            policy_ratio = {agent_id: torch.exp(new_lp_chosens[agent_id] - proxi_lp_chosens[agent_id])
                            for agent_id in self.ids}
            imp_weight = {agent_id: torch.exp(proxi_lp_chosens[agent_id] - behav_lp_chosens[agent_id])
                          for agent_id in self.ids}
            new_Vs = {agent_id: self.critic_nets[agent_id](agent_obs[agent_id])
                      for agent_id in self.ids}
            critic_loss, critic_metrics = self._update_critic(Gs, new_Vs, imp_weight, frozen=critic_frozen)
            actor_loss, actor_metrics = self._update_actor(As, policy_ratio, new_entropies, imp_weight, frozen=actor_frozen)
            total_loss = critic_loss + actor_loss
            total_loss.backward()
            self.optim.step()
            if not critic_frozen:
                self.total_critic_updates += 1
                for agent_id in self.ids:
                    for i, p in enumerate(self.critic_nets[agent_id].parameters()):
                        critic_metrics[agent_id][f"critic_grad_{i}"] = p.grad.detach().norm().item()
            if not actor_frozen:
                self.total_actor_updates += 1
                for agent_id in self.ids:
                    for i, p in enumerate(self.actor_nets[agent_id].parameters()):
                        actor_metrics[agent_id][f"actor_grad_{i}"] = p.grad.detach().norm().item()

        # return
        train_metrics = {
            **critic_metrics,
            **actor_metrics
            }
        return train_metrics

    def _update_critic(self, G, V, imp_weight, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            critic_metrics = {}
            for agent_id in self.ids:
                critic_loss, critic_metrics[agent_id] = self._update_critic(G[agent_id], V[agent_id],
                                                                            imp_weight[agent_id],
                                                                            agent_id=agent_id,
                                                                            frozen=frozen)
                total_loss += critic_loss
            return total_loss, critic_metrics
        else:
            if frozen:
                with torch.no_grad():
                    critic_loss = (G-V).pow(2).mul(imp_weight).mean()
            else:
                #critic_loss = torch.nn.functional.mse_loss(G, V)
                critic_loss = (G-V).pow(2).mul(imp_weight).mean()
                if self.clip_grad.get("critic"):
                    torch.nn.utils.clip_grad_norm_(self.critic_nets[agent_id].parameters(), self.clip_grad["critic"])
            critic_metrics = {
                "critic_loss": critic_loss.item()
                }
            return critic_loss, critic_metrics

    def _update_actor(self, A, ratio, entropy, imp_weight, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A[agent_id],
                                                                         ratio[agent_id],
                                                                         entropy[agent_id],
                                                                         imp_weight[agent_id],
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            A_norm = (A-A.mean())/(A.std() + 1e-8)
            if frozen:
                with torch.no_grad():
                    policy_loss = imp_weight.mul(torch.min(
                        A_norm * ratio,
                        A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                        )).mean()
                    actor_loss = -(policy_loss + self.ent_coef*entropy.mean())
            else:
                policy_loss = imp_weight.mul(torch.min(
                    A_norm * ratio,
                    A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                    )).mean()
                actor_loss = -(policy_loss + self.ent_coef*entropy.mean())
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_nets[agent_id].parameters(), self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy.mean().item(),
                "imp_weight/max": imp_weight.max().item(),
                "imp_weight/min": imp_weight.min().item(),
                "imp_weight/mean": imp_weight.mean().item(),
                }
            return actor_loss, actor_metrics

    @property
    def info(self):
        return dict(
            type="DeIPPO",
            train_steps=self.c_update*self.n_train_envs,
            actor_train_updates=self.total_actor_updates,
            critic_train_updates=self.total_critic_updates,
            actor_learning_rates=self.actor_learning_rates,
            critic_learning_rates=self.critic_learning_rates,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_epochs=self.n_epochs,
            clip_coef=self.clip_coef,
            ent_coef=self.ent_coef,
            clip_grad={"actor": self.clip_grad["actor"], "critic": self.clip_grad["critic"]},
            n_steps=self.n_steps,
            exp_buffer_size=self.exp_buffer_size,
            exp_buffer_replacement_prob=self.exp_buffer_replacement_prob,
            ignore_trunc=self.ignore_trunc
            )


class MAPPO(MultiAgent):
    def __init__(
            self,
            ids=[],
            action_spaces={},
            obs_spaces={},
            joint_space=None,
            actor_hidden_sizes={},
            critic_hidden_size=None,
            actor_learning_rates={},
            critic_learning_rate=None,
            gamma=1.0,
            gae_lambda=1.0,
            n_epochs=1,
            clip_coef=0.1,
            ent_coef=0.0,
            clip_grad={},
            n_steps=5,
            exp_buffer_size=0,
            exp_buffer_replacement_prob=1.0,
            save_to_zoo=False,
            freeze_critic=0,
            freeze_actor=0,
            n_train_envs=1,
            ignore_trunc=True,
        ):
        super().__init__(
            ids,
            action_spaces,
            obs_spaces,
            joint_space
            )
        state_size = list(obs_spaces.values())[0].shape[0]
        action_size = list(action_spaces.values())[0].n
        self.action_size = action_size

        params = []

        # Initialise the decentralised actors
        self.actor_nets = {}
        self.actor_learning_rates = {}
        for agent_id in self.ids:
            self.actor_nets[agent_id] = fc_network([
                state_size,
                *actor_hidden_sizes[agent_id],
                action_size
                ])
            self.actor_learning_rates[agent_id] = actor_learning_rates[agent_id]
            params.append({
                "params": self.actor_nets[agent_id].parameters(),
                "lr": self.actor_learning_rates[agent_id],
                })

        # Initialise the multi-agent critic
        self.critic_net = fc_network([
            self.joint_space.shape[0],
            *critic_hidden_size,
            1
            ])
        self.critic_learning_rate = critic_learning_rate
        params.append({
            "params": self.critic_net.parameters(),
            "lr": self.critic_learning_rate,
            })

        self.optim = Adam(params,
                          lr=params[0]["lr"],
                          eps=1e-5)

        # Initialise hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.n_steps = n_steps
        self.ignore_trunc = ignore_trunc
        self.clip_grad = defaultdict(lambda: False) 
        self.clip_grad.update(clip_grad)
        self.n_train_envs = n_train_envs
        self.freeze_critic = freeze_critic/self.n_train_envs
        self.freeze_actor = freeze_actor/self.n_train_envs
        # divide by n_train_envs since each call to update has n_train_envs transitions

        # Initialise update counters
        self.c_update = 0
        self.p_update = self.n_steps
        self.total_critic_updates = 0
        self.total_actor_updates = 0

        # Initialise memory
        self.last_transition = None
        self.last_n_transitions = ReplayBuffer(n_steps)
        self.exp_buffer_size = exp_buffer_size
        self.exp_buffer_replacement_prob = exp_buffer_replacement_prob
        self.exp_buffer = []

        self.saveables.update({
            "critic": self.critic_net,
            "actor": self.actor_nets,
            "optim": self.optim,
            })
        self.save_to_zoo = save_to_zoo

    def policy(self, obs, explore=True, agent_id=None):
        with torch.no_grad():
            if agent_id is None:
                p = {agent_id: self.policy(obs[agent_id], explore=explore, agent_id=agent_id)
                     for agent_id in self.ids}
            else:
                logits = self.actor_nets[agent_id](torch.Tensor(obs[agent_id]))
                p = torch.distributions.categorical.Categorical(logits=logits)
        return p

    def act(self, obs, explore=True, agent_id=None):
        if agent_id is None:
            return {agent_id: self.act(obs, explore=explore, agent_id=agent_id)
                    for agent_id in self.ids}
        else:
            policy = self.policy(obs, explore=explore, agent_id=agent_id)
            return policy.sample().unsqueeze(1).numpy()

    def update(self):
        self.c_update += 1
        critic_frozen = self.c_update <= self.freeze_critic
        actor_frozen = self.c_update <= self.freeze_actor
        if (critic_frozen and actor_frozen) or self.c_update % self.p_update != 0:
            return {}
        # generate G, V, lp_chosen, entropy
        joint_obs, agent_obs, actions = self._stack_transitions()
        A, G, V = self._compute_gae(joint_obs)
        # update
        with torch.no_grad():
            orig_lp_chosens, _ = self._compute_policy_qty(agent_obs, actions)
        # update loop:
        for epoch in range(self.n_epochs):
            self.optim.zero_grad()
            new_lp_chosens, new_entropies = self._compute_policy_qty(agent_obs, actions)
            policy_ratio = {agent_id: torch.exp(new_lp_chosens[agent_id] - orig_lp_chosens[agent_id])
                            for agent_id in self.ids}
            new_V = self.critic_net(joint_obs)
            critic_loss, critic_metrics = self._update_critic(G, new_V, frozen=critic_frozen)
            actor_loss, actor_metrics = self._update_actor(A, policy_ratio, new_entropies, frozen=actor_frozen)
            total_loss = critic_loss + actor_loss
            total_loss.backward()
            self.optim.step()
            if not critic_frozen:
                self.total_critic_updates += 1
                for i, p in enumerate(self.critic_net.parameters()):
                    critic_metrics[f"critic_grad_{i}"] = p.grad.detach().norm().item()
            if not actor_frozen:
                self.total_actor_updates += 1
                for agent_id in self.ids:
                    for i, p in enumerate(self.actor_nets[agent_id].parameters()):
                        actor_metrics[agent_id][f"actor_grad_{i}"] = p.grad.detach().norm().item()

        # return
        train_metrics = {
            "central_critic": {
                "nTD": A.mean().item(),
                **critic_metrics
                },
            **actor_metrics
            }
        return train_metrics

    def _compute_gae(self, obs):
        V = self.critic_net(obs)
        with torch.no_grad():
            G = torch.zeros_like(V)
            A = torch.zeros_like(V)
            V_n = self.critic_net(self.last_transition.n_obs)
            G_n = V_n.clone()
            A_n = 0
            for i, t in enumerate(reversed(self.last_n_transitions)):
                term = t.terminated.unsqueeze(1)
                trunc = t.truncated.unsqueeze(1)
                r = t.reward.unsqueeze(1)
                if self.ignore_trunc:
                    V_n = ~(term|trunc) * V_n
                    G_n = ~(term|trunc) * G_n
                else:
                    V_n = (~trunc * (V_n * ~term)
                           + trunc * (V_n))
                    G_n = (~trunc * (G_n * ~term)
                           + trunc * (G_n))
                d = r + self.gamma * V_n - V[-i-1, :, :]
                A[-i-1, :, :] = d + self.gamma * self.gae_lambda * A_n
                G[-i-1, :, :] = r + self.gamma * G_n
                A_n = A[-i-1, :, :]
                G_n = G[-i-1, :, :]
                V_n = V[-i-1, :, :]
        return A, G, V

    def _compute_policy_qty(self, obs, acts, agent_id=None):
        if agent_id is None:
            lp_chosens, entropies = zip(*(self._compute_policy_qty(obs, acts, agent_id)
                                          for agent_id in self.ids))
            lp_chosens = dict(zip(self.ids, lp_chosens))
            entropies = dict(zip(self.ids, entropies))
            return lp_chosens, entropies
        else:
            logits = self.actor_nets[agent_id](obs[agent_id])
            lp = torch.nn.functional.log_softmax(logits, dim=2)
            lp_chosen = lp.gather(2, acts[agent_id])
            entropy = -lp.exp().mul(lp).sum(dim=2).unsqueeze(2)
            return lp_chosen, entropy

    def _update_critic(self, G, V, frozen=False):
        if frozen:
            with torch.no_grad():
                critic_loss = torch.nn.functional.mse_loss(G, V)
        else:
            critic_loss = torch.nn.functional.mse_loss(G, V)
            if self.clip_grad.get("critic"):
                torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.clip_grad["critic"])

        critic_metrics = {
            "critic_loss": critic_loss.item()
            }
        return critic_loss, critic_metrics

    def _update_actor(self, A, ratio, entropy, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A,
                                                                         ratio[agent_id],
                                                                         entropy[agent_id],
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            A_norm = (A-A.mean())/(A.std() + 1e-8)
            if frozen:
                with torch.no_grad():
                    policy_loss = torch.min(
                        A_norm * ratio,
                        A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                        ).mean()
                    actor_loss = -(policy_loss + self.ent_coef*entropy.mean())
            else:
                policy_loss = torch.min(
                    A_norm * ratio,
                    A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                    ).mean()
                actor_loss = -(policy_loss + self.ent_coef*entropy.mean())
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_nets[agent_id].parameters(), self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy.mean().item(),
                }
            return actor_loss, actor_metrics

    def store_transition(self, transitions, agent_id=None):
        # I expect this to come in as a dict of individual agent Transitions
        obs = {agent_id: torch.FloatTensor(t.obs)
               for agent_id, t in transitions.items()}
        action = {agent_id: torch.LongTensor(t.action)
                  for agent_id, t in transitions.items()}
        # Assumes common-payoff
        reward = torch.FloatTensor(list(transitions.values())[0].reward)
        terminated = torch.BoolTensor(list(transitions.values())[0].terminated)
        truncated = torch.BoolTensor(list(transitions.values())[0].truncated)
        joint_obs = torch.FloatTensor(list(transitions.values())[0].obs)
        n_obs = torch.FloatTensor(list(transitions.values())[0].n_obs)
        transition = Transition(
            obs=obs,
            action=action,
            n_obs=n_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            joint_obs=joint_obs,
            )
        self.last_transition = transition
        self.last_n_transitions.push(transition)
        self.store_exp(joint_obs)

    def _stack_transitions(self):
        joint_obs = torch.stack(list(t.joint_obs for t in self.last_n_transitions))
        agent_obs = {agent_id: torch.stack(list(t.obs[agent_id] for t in self.last_n_transitions))
                     for agent_id in self.ids}
        actions = {agent_id: torch.stack(list(t.action[agent_id] for t in self.last_n_transitions))
                   for agent_id in self.ids}
        return joint_obs, agent_obs, actions

    def store_exp(self, obs):
        if self.exp_buffer_size == 0:
            pass
        elif len(self.exp_buffer) >= self.exp_buffer_size:
            if np.random.random() < self.exp_buffer_replacement_prob:
                idx = np.random.randint(self.exp_buffer_size)
                self.exp_buffer[idx] = obs
        else:
            self.exp_buffer.append(obs)

    def add_agent(
            self,
            agent_id,
            action_space: gym.Space,
            obs_space: gym.Space,
            actor_hidden_size,
            critic_hidden_size,
            actor_learning_rate,
            critic_learning_rate,
            gamma,
            ent_coef,
            n_steps,
            save_to_zoo=False
            ):
        ...

    @property
    def info(self):
        return dict(
            type="MAPPO",
            train_steps=self.c_update*self.n_train_envs,
            actor_train_updates=self.total_actor_updates,
            critic_train_updates=self.total_critic_updates,
            actor_learning_rates=self.actor_learning_rates,
            critic_learning_rate=self.critic_learning_rate,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_epochs=self.n_epochs,
            clip_coef=self.clip_coef,
            ent_coef=self.ent_coef,
            clip_grad={"actor": self.clip_grad["actor"], "critic": self.clip_grad["critic"]},
            n_steps=self.n_steps,
            exp_buffer_size=self.exp_buffer_size,
            exp_buffer_replacement_prob=self.exp_buffer_replacement_prob,
            ignore_trunc=self.ignore_trunc
            )

    def zoo_save(self, zoo_path, agent_save_names, env_cfg, evals, seed):
        agent_save_name_mapping = dict(zip(self.ids, agent_save_names))
        concat_name = "_".join(agent_save_names)
        # configs
        # actors
        # inelegant way of changing the evals into floats (from numpy.floats)
        # required so omegaconf can read
        float_evals = {
                "team": {k: v.item() for k, v in evals["team"].items()},
                "individual": {k: {k: v.item() for k, v in v.items()}
                               for k, v in evals["individual"].items()},
                }
        info = self.info
        for agent_id in self.ids:
            agent_name = agent_save_name_mapping[agent_id]
            model_pathname = osp.normpath(
                osp.join(zoo_path, "outputs", "actors", f"{agent_name}.pt")
                )
            config_pathname = osp.normpath(
                osp.join(zoo_path, "configs", "actors", f"{agent_name}.yaml")
                )
            os.makedirs(osp.dirname(model_pathname), exist_ok=True)
            os.makedirs(osp.dirname(config_pathname), exist_ok=True)
            torch.save(self.actor_nets[agent_id], model_pathname)
            partner_agents = [agent_save_name_mapping[other_agent_id]
                              for other_agent_id in self.ids
                              if other_agent_id != agent_id]
            model_dict = dict(
                internal_name=agent_id,
                train_steps=info["train_steps"],
                train_updates=info["actor_train_updates"],
                learning_rate=info["actor_learning_rates"][agent_id],
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_epochs=self.n_epochs,
                clip_coef=self.clip_coef,
                ent_coef=self.ent_coef,
                clip_grad=self.clip_grad["actor"],
                n_steps=self.n_steps,
                ignore_trunc=self.ignore_trunc,
                )
            actor_dict = dict(
                name=agent_name,
                model=model_dict,
                train_env=env_cfg,
                eval={
                    "team": float_evals["team"],
                    "individual": float_evals["individual"][agent_id],
                    },
                critic=concat_name,
                experience=concat_name,
                partner_agents=partner_agents,
                path_to_model=model_pathname,
                seed=seed,
                )
            actor_dict = OmegaConf.create(actor_dict)
            with open(config_pathname, "w") as f:
                f.write(OmegaConf.to_yaml(actor_dict))

        # critic
        model_pathname = osp.normpath(
            osp.join(zoo_path, "outputs", "critics", f"{concat_name}.pt")
            )
        config_pathname = osp.normpath(
            osp.join(zoo_path, "configs", "critics", f"{concat_name}.yaml")
            )
        os.makedirs(osp.dirname(model_pathname), exist_ok=True)
        os.makedirs(osp.dirname(config_pathname), exist_ok=True)
        torch.save(self.critic_net, model_pathname)
        model_dict = dict(
            train_steps=info["train_steps"],
            train_updates=info["critic_train_updates"],
            learning_rate=info["critic_learning_rate"],
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_epochs=self.n_epochs,
            clip_coef=self.clip_coef,
            clip_grad=self.clip_grad["critic"],
            n_steps=self.n_steps,
            ignore_trunc=self.ignore_trunc,
            )
        critic_dict = dict(
            name=concat_name,
            model=model_dict,
            train_env=env_cfg,
            eval=float_evals["team"],
            actors=agent_save_names,
            experience=concat_name,
            path_to_model=model_pathname,
            seed=seed,
            )
        critic_dict = OmegaConf.create(critic_dict)
        with open(config_pathname, "w") as f:
            f.write(OmegaConf.to_yaml(critic_dict))
        # experience
        experience_pathname = osp.normpath(
            osp.join(zoo_path, "outputs", "experience", f"{concat_name}.pt")
            )
        config_pathname = osp.normpath(
            osp.join(zoo_path, "configs", "experience", f"{concat_name}.yaml")
            )
        os.makedirs(osp.dirname(experience_pathname), exist_ok=True)
        os.makedirs(osp.dirname(config_pathname), exist_ok=True)
        if len(self.exp_buffer) > 0:
            save_buffer = torch.concat(self.exp_buffer)
        else:
            save_buffer = []
        torch.save(save_buffer, experience_pathname)
        buffer_dict = dict(
            max_size=self.exp_buffer_size,
            replacement_prob=self.exp_buffer_replacement_prob,
            size=len(self.exp_buffer),
            samples=len(save_buffer),
                )
        experience_dict = dict(
            name=concat_name,
            train_env=env_cfg,
            eval=float_evals["team"],
            experience=buffer_dict,
            critic=concat_name,
            actors=agent_save_names,
            path_to_experience=experience_pathname,
            seed=seed,
            )
        experience_dict = OmegaConf.create(experience_dict)
        with open(config_pathname, "w") as f:
            f.write(OmegaConf.to_yaml(experience_dict))
        return agent_save_name_mapping

    @classmethod
    def from_networks(
            cls,
            networks={},
            action_spaces={},
            obs_spaces={},
            joint_space=None,
            actor_learning_rates={},
            critic_learning_rate=None,
            gamma=1.0,
            gae_lambda=1.0,
            n_epochs=1,
            clip_coef=0.1,
            ent_coef=0.0,
            n_steps=5,
            exp_buffer_size=1e4,
            exp_buffer_replacement_prob=1e-2,
            save_to_zoo=False,
            freeze_critic=0,
            freeze_actor=0,
            n_train_envs=1,
            ignore_trunc=True,
        ):
        ids = [agent_id for agent_id in networks.keys() if agent_id != "central_critic"]
        agent = cls(
            ids=ids,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_hidden_sizes={agent_id: [1] for agent_id in ids},
            critic_hidden_size=[1],
            actor_learning_rates=actor_learning_rates,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_epochs=n_epochs,
            clip_coef=clip_coef,
            ent_coef=ent_coef,
            n_steps=n_steps,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=save_to_zoo,
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=ignore_trunc,
            )

        # update the networks
        params = []
        for agent_id in ids:
            agent.actor_nets[agent_id] = copy(networks[agent_id])
            params.append({
                "params": agent.actor_nets[agent_id].parameters(),
                "lr": actor_learning_rates[agent_id],
                })
        agent.critic_net = copy(networks["central_critic"])
        params.append({
            "params": agent.critic_net.parameters(),
            "lr": critic_learning_rate,
            })
        agent.optim = Adam(params,
                          lr=params[0]["lr"],
                          eps=1e-5)
        agent.saveables.update({
            "critic": agent.critic_net,
            "actor": agent.actor_nets,
            "optim": agent.optim,
            })
        return agent

    @classmethod
    def from_config(cls, cfg, env):
        agent_ids = [a["agent_id"] for a in cfg.agents]
        assert len(agent_ids) == cfg.n_agents, \
            f"len(agent_ids) = {len(agent_ids)} (expected {cfg.n_agents})"
        action_spaces = {
            agent_id: env.action_space
            for agent_id in agent_ids
            }
        obs_spaces = {
            agent_id: env.observation_space
            for agent_id in agent_ids
            }
        if hasattr(env, "joint_space"):
            joint_space = env.joint_space
        else:
            joint_space = env.observation_space
        actor_hidden_sizes = {
            agent_id: cfg.model.actor
            for agent_id in agent_ids
            }
        critic_hidden_size = cfg.model.critic
        actor_learning_rates = {
            agent_id: cfg.lr.actor
            for agent_id in agent_ids
            }
        critic_learning_rate = cfg.lr.critic
        gamma = cfg.get("gamma", 1.0)
        gae_lambda = cfg.get("gae_lambda", 1)
        n_epochs = cfg.get("n_epochs", 1)
        clip_coef = cfg.get("clip_coef", 0.1)
        ent_coef = cfg.get("ent_coef", 0.0)
        n_steps = cfg.get("n_steps", 5)
        clip_grad = cfg.get("clip_grad", {})
        save_to_zoo = cfg.get("save_to_zoo", False)
        ignore_trunc = cfg.get("ignore_trunc", True)
        if "exp_buffer" in cfg.keys():
            exp_buffer_size = cfg.exp_buffer.get("size", 0)
            exp_buffer_replacement_prob = cfg.exp_buffer.get("replacement_prob", 0.0)
        else:
            exp_buffer_size = 0
            exp_buffer_replacement_prob = 1.0

        if "freeze" in cfg.keys():
            freeze_critic = cfg.freeze.get("critic", 0)
            freeze_actor = cfg.freeze.get("actor", 0)
        else:
            freeze_critic = 0
            freeze_actor = 0

        if hasattr(env, "num_vec_envs"):
            n_train_envs = env.num_vec_envs
        elif hasattr(env, "vec_envs"):
            n_train_envs = len(env.vec_envs)
        else:
            n_train_envs = 1

        return cls(
            ids=agent_ids,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_hidden_sizes=actor_hidden_sizes,
            critic_hidden_size=critic_hidden_size,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_epochs=n_epochs,
            clip_coef=clip_coef,
            ent_coef=ent_coef,
            n_steps=n_steps,
            clip_grad=clip_grad,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=save_to_zoo,
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=ignore_trunc,
            )

    @classmethod
    def from_zoo(cls, actor_names, env, zoo_path, override_cfg={}, save_to_zoo=False):
        networks = {}
        critic_names = set()
        name_id_mapping = {}
        agent_ids = []
        actor_learning_rates = {}

        for actor_name in actor_names:
            # load config
            path = osp.normpath(
                osp.join(zoo_path, "configs", "actors", f"{actor_name}.yaml")
                )
            cfg = OmegaConf.load(path)
            agent_id = cfg.model.internal_name
            agent_ids.append(agent_id)
            name_id_mapping[actor_name] = agent_id
            critic_names.add(cfg.critic)
            actor_learning_rates[agent_id] = cfg.model.learning_rate
            ent_coef = cfg.model.ent_coef
            networks[agent_id] = torch.load(cfg.path_to_model)


        if len(cfg.partner_agents) + 1 < len(actor_names):
            raise Exception("MAA2C.from_zoo method got fewer actors than expected")

        if len(critic_names) != 1:
            raise Exception(
                f"""MAA2C.from_zoo method only applies to agents with a shared critic.
                Got critics {critic_names}.
                """
                )

        # Load env
        action_spaces = {
            agent_id: env.action_space
            for agent_id in agent_ids
            }
        obs_spaces = {
            agent_id: env.observation_space
            for agent_id in agent_ids
            }

        if hasattr(env, "joint_space"):
            joint_space = env.joint_space
        else:
            joint_space = env.observation_space

        # Load critic
        critic_name = critic_names.pop()
        path = osp.normpath(
            osp.join(zoo_path, "configs", "critics", f"{critic_name}.yaml")
            )
        cfg = OmegaConf.load(path)
        critic_learning_rate = cfg.model.learning_rate
        networks["central_critic"] = torch.load(cfg.path_to_model)
        ignore_trunc = cfg.model.get("ignore_trunc", True)

        actor_learning_rates = override_cfg.get("actor_learning_rates", actor_learning_rates)
        critic_learning_rate = override_cfg.get("critic_learning_rate", critic_learning_rate)
        gamma = override_cfg.get("gamma", cfg.model.gamma)
        ent_coef = override_cfg.get("ent_coef", ent_coef)
        n_steps = override_cfg.get("n_steps", cfg.model.n_steps)
        ignore_trunc = override_cfg.get("ignore_trunc", ignore_trunc)
        # Doesn't make sense to load these from the zoo
        freeze_critic = override_cfg.get("freeze_critic", 0)
        freeze_actor = override_cfg.get("freeze_actor", 0)

        # Load experience config
        path = osp.normpath(
            osp.join(zoo_path, "configs", "experience", f"{cfg.experience}.yaml")
            )
        cfg = OmegaConf.load(path)
        exp_buffer_size = override_cfg.get("exp_buffer_size", cfg.experience.max_size)
        exp_buffer_replacement_prob = \
            override_cfg.get("exp_buffer_replacement_prob", cfg.experience.replacement_prob)

        if hasattr(env, "num_vec_envs"):
            n_train_envs = env.num_vec_envs
        elif hasattr(env, "vec_envs"):
            n_train_envs = len(env.vec_envs)
        else:
            n_train_envs = 1

        agents = MAPPO.from_networks(
            networks=networks,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            gae_lambda=cfg.gae_lambda,
            n_epochs=cfg.n_epochs,
            clip_coef=cfg.clip_coef,
            ent_coef=ent_coef,
            n_steps=n_steps,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=save_to_zoo,
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=ignore_trunc,
            )
        return agents


class TabularIA2C(MultiAgent):
    def __init__(
            self,
            ids=[],
            action_spaces={},
            obs_spaces={},
            joint_space=None,
            actor_learning_rates={},
            critic_learning_rates={},
            gamma=1.0,
            ent_coef=0.0,
            clip_grad={},
            n_steps=5,
            exp_buffer_size=0,
            exp_buffer_replacement_prob=1.0,
            save_to_zoo=False,
            freeze_critic=0,
            freeze_actor=0,
            n_train_envs=1,
            ignore_trunc=True,
        ):
        super().__init__(
            ids,
            action_spaces,
            obs_spaces,
            joint_space
            )
        #state_size = obs_space.shape[1]  # When the parallel envs are in the Space
        state_size = list(obs_spaces.values())[0].shape[0]
        action_size = list(action_spaces.values())[0].n
        self.action_size = action_size

        params = []

        # Initialise the decentralised actors & critics
        self.actor_tabs = {}
        self.actor_learning_rates = {}
        self.critic_tabs = {}
        self.critic_learning_rates = {}
        for agent_id in self.ids:
            self.actor_tabs[agent_id] = torch.zeros(state_size, action_size, requires_grad=True)
            self.actor_learning_rates[agent_id] = actor_learning_rates[agent_id]
            self.critic_tabs[agent_id] = torch.zeros(state_size, requires_grad=True)
            self.critic_learning_rates[agent_id] = critic_learning_rates[agent_id]
            params.append({
                "params": self.actor_tabs[agent_id],
                "lr": self.actor_learning_rates[agent_id],
                })
            params.append({
                "params": self.critic_tabs[agent_id],
                "lr": self.critic_learning_rates[agent_id],
                })
        self.optim = Adam(params,
                          lr=params[0]["lr"],
                          eps=1e-5)

        # Initialise hyperparameters
        self.gamma = gamma
        self.gae_lambda = 1
        self.ent_coef = ent_coef
        self.n_steps = n_steps
        self.ignore_trunc = ignore_trunc
        self.clip_grad = defaultdict(lambda: False) 
        self.clip_grad.update(clip_grad)
        self.n_train_envs = n_train_envs
        self.freeze_critic = freeze_critic/self.n_train_envs
        self.freeze_actor = freeze_actor/self.n_train_envs
        # divide by n_train_envs since each call to update has n_train_envs transitions

        # Initialise update counters
        self.c_update = 0
        self.p_update = self.n_steps
        self.total_critic_updates = 0
        self.total_actor_updates = 0

        # Initialise memory
        self.last_transition = None
        self.last_n_transitions = ReplayBuffer(n_steps)
        self.exp_buffer_size = exp_buffer_size
        self.exp_buffer_replacement_prob = exp_buffer_replacement_prob
        self.exp_buffer = {agent_id: [] for agent_id in self.ids}

        self.saveables.update({
            "critic": self.critic_tabs,
            "actor": self.actor_tabs,
            "optim": self.optim,
            })
        self.save_to_zoo = save_to_zoo

    def policy(self, obs, explore=True, agent_id=None):
        with torch.no_grad():
            if agent_id is None:
                p = {agent_id: self.policy(obs[agent_id], explore=explore, agent_id=agent_id)
                     for agent_id in self.ids}
            else:
                obs_torch = torch.Tensor(obs[agent_id])
                logits = obs_torch.matmul(self.actor_tabs[agent_id])
                p = torch.distributions.categorical.Categorical(logits=logits)
        return p

    def act(self, obs, explore=True, agent_id=None):
        if agent_id is None:
            return {agent_id: self.act(obs, explore=explore, agent_id=agent_id)
                    for agent_id in self.ids}
        else:
            policy = self.policy(obs, explore=explore, agent_id=agent_id)
            return policy.sample().unsqueeze(1).numpy()

    def update(self):
        self.c_update += 1
        critic_frozen = self.c_update <= self.freeze_critic
        actor_frozen = self.c_update <= self.freeze_actor
        if (critic_frozen and actor_frozen) or self.c_update % self.p_update != 0:
            return {}
        # generate G, V, lp_chosen, entropy
        _, agent_obs, actions = self._stack_transitions()
        As, Gs, Vs = self._compute_gae(agent_obs)
        lp_chosens, entropies = self._compute_policy_qty(agent_obs, actions)
        # update
        self.optim.zero_grad()
        critic_loss, critic_metrics = self._update_critic(Gs, Vs, frozen=critic_frozen)
        actor_loss, actor_metrics = self._update_actor(As, lp_chosens, entropies, frozen=actor_frozen)
        total_loss = critic_loss + actor_loss
        total_loss.backward()
        if not critic_frozen:
            self.total_critic_updates += 1
            for agent_id in self.ids:
                critic_metrics[agent_id][f"critic_grad"] = self.critic_tabs[agent_id].grad.detach().norm().item()
        if not actor_frozen:
            self.total_actor_updates += 1
            for agent_id in self.ids:
                actor_metrics[agent_id][f"actor_grad"] = self.actor_tabs[agent_id].grad.detach().norm().item()
        self.optim.step()

        # return
        train_metrics = {
            **critic_metrics,
            **actor_metrics
            }
        return train_metrics

    def _compute_gae(self, obs, agent_id=None):
        if agent_id is None:
            As, Gs, Vs = zip(*(self._compute_gae(obs, agent_id=agent_id)
                           for agent_id in self.ids))
            As = dict(zip(self.ids, As))
            Gs = dict(zip(self.ids, Gs))
            Vs = dict(zip(self.ids, Vs))
            return As, Gs, Vs
        else:
            V = obs[agent_id].matmul(self.critic_tabs[agent_id]).unsqueeze(-1)
            with torch.no_grad():
                G = torch.zeros_like(V)
                A = torch.zeros_like(V)
                V_n = self.last_transition.n_obs[agent_id].matmul(self.critic_tabs[agent_id]).unsqueeze(-1)
                G_n = V_n.clone()
                A_n = 0
                for i, t in enumerate(reversed(self.last_n_transitions)):
                    term = t.terminated.unsqueeze(1)
                    trunc = t.truncated.unsqueeze(1)
                    r = t.reward.unsqueeze(1)
                    if self.ignore_trunc:
                        V_n = ~(term|trunc) * V_n
                        G_n = ~(term|trunc) * G_n
                    else:
                        V_n = (~trunc * (V_n * ~term)
                               + trunc * (V_n))
                        G_n = (~trunc * (G_n * ~term)
                               + trunc * (G_n))
                    d = r + self.gamma * V_n - V[-i-1, :, :]
                    A[-i-1, :, :] = d + self.gamma * self.gae_lambda * A_n
                    G[-i-1, :, :] = r + self.gamma * G_n
                    A_n = A[-i-1, :, :]
                    G_n = G[-i-1, :, :]
                    V_n = V[-i-1, :, :]
            return A, G, V

    def _compute_policy_qty(self, obs, acts, agent_id=None):
        if agent_id is None:
            lp_chosens, entropies = zip(*(self._compute_policy_qty(obs, acts, agent_id)
                                          for agent_id in self.ids))
            lp_chosens = dict(zip(self.ids, lp_chosens))
            entropies = dict(zip(self.ids, entropies))
            return lp_chosens, entropies
        else:
            logits = obs[agent_id].matmul(self.actor_tabs[agent_id])
            lp = torch.nn.functional.log_softmax(logits, dim=2)
            lp_chosen = lp.gather(2, acts[agent_id])
            entropy = -lp.exp().mul(lp).sum(dim=2).unsqueeze(2)
            return lp_chosen, entropy

    def _update_critic(self, G, V, frozen=False, agent_id=None):
        if agent_id is None:
            total_loss = 0
            critic_metrics = {}
            for agent_id in self.ids:
                critic_loss, critic_metrics[agent_id] = self._update_critic(G[agent_id], V[agent_id],
                                                                            agent_id=agent_id,
                                                                            frozen=frozen)
                total_loss += critic_loss
            return total_loss, critic_metrics
        else:
            if frozen:
                with torch.no_grad():
                    critic_loss = torch.nn.functional.mse_loss(G, V)
            else:
                critic_loss = torch.nn.functional.mse_loss(G, V)
                if self.clip_grad.get("critic"):
                    torch.nn.utils.clip_grad_norm_(self.critic_tabs[agent_id], self.clip_grad["critic"])
            critic_metrics = {
                "critic_loss": critic_loss.item()
                }
            return critic_loss, critic_metrics

    def _update_actor(self, A, lp_chosen, entropy, weights=1.0, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A[agent_id],
                                                                         lp_chosen,
                                                                         entropy,
                                                                         weights,
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            if frozen:
                with torch.no_grad():
                    actor_loss = - (lp_chosen[agent_id].mul(A)
                                    + self.ent_coef*entropy[agent_id]).mul(weights).mean()
            else:
                actor_loss = - (lp_chosen[agent_id].mul(A)
                                + self.ent_coef*entropy[agent_id]).mul(weights).mean()
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_tabs[agent_id], self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy[agent_id].mean().item(),
                }
            return actor_loss, actor_metrics

    def store_transition(self, transitions, agent_id=None):
        # I expect this to come in as a dict of individual agent Transitions
        obs = {agent_id: torch.FloatTensor(t.obs)
               for agent_id, t in transitions.items()}
        n_obs = {agent_id: torch.FloatTensor(t.n_obs)
               for agent_id, t in transitions.items()}
        action = {agent_id: torch.LongTensor(t.action)
                  for agent_id, t in transitions.items()}
        # Assumes common-payoff
        reward = torch.FloatTensor(list(transitions.values())[0].reward)
        terminated = torch.BoolTensor(list(transitions.values())[0].terminated)
        truncated = torch.BoolTensor(list(transitions.values())[0].truncated)
        joint_obs = torch.FloatTensor(list(transitions.values())[0].obs)
        transition = Transition(
            obs=obs,
            action=action,
            n_obs=n_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            joint_obs=joint_obs,
            )
        self.last_transition = transition
        self.last_n_transitions.push(transition)
        self.store_exp(obs)

    def _stack_transitions(self):
        joint_obs = torch.stack(list(t.joint_obs for t in self.last_n_transitions))
        agent_obs = {agent_id: torch.stack(list(t.obs[agent_id] for t in self.last_n_transitions))
                     for agent_id in self.ids}
        actions = {agent_id: torch.stack(list(t.action[agent_id] for t in self.last_n_transitions))
                   for agent_id in self.ids}
        return joint_obs, agent_obs, actions

    def store_exp(self, obs, agent_id=None):
        if agent_id is None:
            for agent_id in self.ids:
                self.store_exp(obs, agent_id=agent_id)
        else:
            if self.exp_buffer_size == 0:
                pass
            elif len(self.exp_buffer[agent_id]) >= self.exp_buffer_size:
                if np.random.random() < self.exp_buffer_replacement_prob:
                    idx = np.random.randint(self.exp_buffer_size)
                    self.exp_buffer[idx] = obs[agent_id]
            else:
                self.exp_buffer[agent_id].append(obs[agent_id])

    def add_agent(
            self,
            agent_id,
            action_space: gym.Space,
            obs_space: gym.Space,
            actor_hidden_size,
            critic_hidden_size,
            actor_learning_rate,
            critic_learning_rate,
            gamma,
            ent_coef,
            n_steps,
            save_to_zoo=False
            ):
        ...

    @property
    def info(self):
        return dict(
            type="TabularIA2C",
            train_steps=self.c_update*self.n_train_envs,
            actor_train_updates=self.total_actor_updates,
            critic_train_updates=self.total_critic_updates,
            actor_learning_rates=self.actor_learning_rates,
            critic_learning_rates=self.critic_learning_rates,
            gamma=self.gamma,
            ent_coef=self.ent_coef,
            clip_grad={"actor": self.clip_grad["actor"], "critic": self.clip_grad["critic"]},
            n_steps=self.n_steps,
            exp_buffer_size=self.exp_buffer_size,
            exp_buffer_replacement_prob=self.exp_buffer_replacement_prob,
            ignore_trunc=self.ignore_trunc
            )
    ...
    @classmethod
    def from_config(cls, cfg, env):
        agent_ids = [a["agent_id"] for a in cfg.agents]
        assert len(agent_ids) == cfg.n_agents, \
            f"len(agent_ids) = {len(agent_ids)} (expected {cfg.n_agents})"
        action_spaces = {
            agent_id: env.action_space
            for agent_id in agent_ids
            }
        obs_spaces = {
            agent_id: env.observation_space
            for agent_id in agent_ids
            }
        if hasattr(env, "joint_space"):
            joint_space = env.joint_space
        else:
            joint_space = env.observation_space
        actor_learning_rates = {
            agent_id: cfg.lr.actor
            for agent_id in agent_ids
            }
        critic_learning_rates = {
            agent_id: cfg.lr.critic
            for agent_id in agent_ids
            }
        gamma = cfg.get("gamma", 1.0)
        ent_coef = cfg.get("ent_coef", 0.0)
        n_steps = cfg.get("n_steps", 5)
        clip_grad = cfg.get("clip_grad", {})
        save_to_zoo = cfg.get("save_to_zoo", False)
        ignore_trunc = cfg.get("ignore_trunc", True)
        if "exp_buffer" in cfg.keys():
            exp_buffer_size = cfg.exp_buffer.get("size", 0)
            exp_buffer_replacement_prob = cfg.exp_buffer.get("replacement_prob", 0.0)
        else:
            exp_buffer_size = 0
            exp_buffer_replacement_prob = 1.0

        if "freeze" in cfg.keys():
            freeze_critic = cfg.freeze.get("critic", 0)
            freeze_actor = cfg.freeze.get("actor", 0)
        else:
            freeze_critic = 0
            freeze_actor = 0

        if hasattr(env, "num_vec_envs"):
            n_train_envs = env.num_vec_envs
        elif hasattr(env, "vec_envs"):
            n_train_envs = len(env.vec_envs)
        else:
            n_train_envs = 1

        return cls(
            ids=agent_ids,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rates=critic_learning_rates,
            gamma=gamma,
            ent_coef=ent_coef,
            n_steps=n_steps,
            clip_grad=clip_grad,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=save_to_zoo,
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=ignore_trunc,
            )

    def zoo_save(self, zoo_path, agent_save_names, env_cfg, evals, seed):
        agent_save_name_mapping = dict(zip(self.ids, agent_save_names))
        concat_name = "_".join(agent_save_names)
        # configs
        # actors
        # inelegant way of changing the evals into floats (from numpy.floats)
        # required so omegaconf can read
        float_evals = {
                "team": {k: v.item() for k, v in evals["team"].items()},
                "individual": {k: {k: v.item() for k, v in v.items()}
                               for k, v in evals["individual"].items()},
                }
        info = self.info
        for agent_id in self.ids:
            agent_name = agent_save_name_mapping[agent_id]
            partner_agents = [agent_save_name_mapping[other_agent_id]
                              for other_agent_id in self.ids
                              if other_agent_id != agent_id]
            # Actor
            actor_model_pathname = osp.normpath(
                osp.join(zoo_path, "outputs", "actors", f"{agent_name}.pt")
                )
            actor_config_pathname = osp.normpath(
                osp.join(zoo_path, "configs", "actors", f"{agent_name}.yaml")
                )
            os.makedirs(osp.dirname(actor_model_pathname), exist_ok=True)
            os.makedirs(osp.dirname(actor_config_pathname), exist_ok=True)
            torch.save(self.actor_tabs[agent_id], actor_model_pathname)
            actor_model_dict = dict(
                internal_name=agent_id,
                train_steps=info["train_steps"],
                train_updates=info["actor_train_updates"],
                learning_rate=info["actor_learning_rates"][agent_id],
                gamma=self.gamma,
                ent_coef=self.ent_coef,
                clip_grad=self.clip_grad["actor"],
                n_steps=self.n_steps,
                ignore_trunc=self.ignore_trunc,
                )
            actor_dict = dict(
                name=agent_name,
                model=actor_model_dict,
                train_env=env_cfg,
                eval={
                    "team": float_evals["team"],
                    "individual": float_evals["individual"][agent_id],
                    },
                critic=agent_name,
                experience=agent_name,
                partner_agents=partner_agents,
                path_to_model=actor_model_pathname,
                seed=seed,
                )
            actor_dict = OmegaConf.create(actor_dict)
            with open(actor_config_pathname, "w") as f:
                f.write(OmegaConf.to_yaml(actor_dict))

            # Critic
            critic_model_pathname = osp.normpath(
                osp.join(zoo_path, "outputs", "critics", f"{agent_name}.pt")
                )
            critic_config_pathname = osp.normpath(
                osp.join(zoo_path, "configs", "critics", f"{agent_name}.yaml")
                )
            os.makedirs(osp.dirname(critic_model_pathname), exist_ok=True)
            os.makedirs(osp.dirname(critic_config_pathname), exist_ok=True)
            torch.save(self.critic_tabs[agent_id], critic_model_pathname)
            critic_model_dict = dict(
                internal_name=agent_id,
                train_steps=info["train_steps"],
                train_updates=info["critic_train_updates"],
                learning_rate=info["critic_learning_rates"][agent_id],
                gamma=self.gamma,
                ent_coef=self.ent_coef,
                clip_grad=self.clip_grad["critic"],
                n_steps=self.n_steps,
                ignore_trunc=self.ignore_trunc,
                )
            critic_dict = dict(
                name=agent_name,
                model=critic_model_dict,
                train_env=env_cfg,
                eval={
                    "team": float_evals["team"],
                    "individual": float_evals["individual"][agent_id],
                    },
                actor=agent_name,
                experience=agent_name,
                partner_agents=partner_agents,
                path_to_model=critic_model_pathname,
                seed=seed,
                )
            critic_dict = OmegaConf.create(critic_dict)
            with open(critic_config_pathname, "w") as f:
                f.write(OmegaConf.to_yaml(critic_dict))

            # experience
            experience_pathname = osp.normpath(
                osp.join(zoo_path, "outputs", "experience", f"{agent_name}.pt")
                )
            config_pathname = osp.normpath(
                osp.join(zoo_path, "configs", "experience", f"{agent_name}.yaml")
                )
            os.makedirs(osp.dirname(experience_pathname), exist_ok=True)
            os.makedirs(osp.dirname(config_pathname), exist_ok=True)
            if len(self.exp_buffer) > 0:
                save_buffer = torch.concat(self.exp_buffer[agent_id])
            else:
                save_buffer = []
            torch.save(save_buffer, experience_pathname)
            buffer_dict = dict(
                max_size=self.exp_buffer_size,
                replacement_prob=self.exp_buffer_replacement_prob,
                size=len(self.exp_buffer),
                samples=len(save_buffer),
                    )
            experience_dict = dict(
                name=agent_name,
                train_env=env_cfg,
                eval=float_evals["team"],
                experience=buffer_dict,
                critics=agent_save_names,
                actors=agent_save_names,
                path_to_experience=experience_pathname,
                seed=seed,
                )
            experience_dict = OmegaConf.create(experience_dict)
            with open(config_pathname, "w") as f:
                f.write(OmegaConf.to_yaml(experience_dict))
        return agent_save_name_mapping

    @classmethod
    def load_from_zoo(cls, cfg, name_id_mapping, env, zoo_path):
        actors = {}
        critics = {}
        agent_ids = list(name_id_mapping.values())
        exp_buffers = []
        for actor_name, agent_id in name_id_mapping.items():
            # load config
            actor_cfg_path = osp.normpath(
                osp.join(zoo_path, "configs", "actors", f"{actor_name}.yaml")
                )
            actor_cfg = OmegaConf.load(actor_cfg_path)
            actors[agent_id] = torch.load(actor_cfg.path_to_model)
            # load critic network
            critic_cfg_path = osp.normpath(
                osp.join(zoo_path, "configs", "critics", f"{actor_cfg.critic}.yaml")
                )
            critic_cfg = OmegaConf.load(critic_cfg_path)
            critics[agent_id] = torch.load(critic_cfg.path_to_model)
            # load experience
            exp_path = osp.normpath(
                osp.join(zoo_path, "configs", "experience", f"{actor_cfg.experience}.yaml")
                )
            exp_cfg = OmegaConf.load(exp_path)
            exp_buffers.append(torch.load(exp_cfg.path_to_experience))

        # Load env
        action_spaces = {
            agent_id: env.action_space
            for agent_id in agent_ids
            }
        obs_spaces = {
            agent_id: env.observation_space
            for agent_id in agent_ids
            }
        if hasattr(env, "joint_space"):
            joint_space = env.joint_space
        else:
            joint_space = env.observation_space

        actor_learning_rates = {
            agent_id: cfg.lr.actor
            for agent_id in agent_ids
            }
        critic_learning_rates = {
            agent_id: cfg.lr.critic
            for agent_id in agent_ids
            }
        if "exp_buffer" in cfg.keys():
            exp_buffer_size = cfg.exp_buffer.get("size", 0)
            exp_buffer_replacement_prob = cfg.exp_buffer.get("replacement_prob", 0.0)
        else:
            exp_buffer_size = 0
            exp_buffer_replacement_prob = 1.0
        if "freeze" in cfg.keys():
            freeze_critic = cfg.freeze.get("critic", 0)
            freeze_actor = cfg.freeze.get("actor", 0)
        else:
            freeze_critic = 0
            freeze_actor = 0

        if hasattr(env, "num_vec_envs"):
            n_train_envs = env.num_vec_envs
        elif hasattr(env, "vec_envs"):
            n_train_envs = len(env.vec_envs)
        else:
            n_train_envs = 1

        agents = TabularIA2C.from_tables(
            actors=actors,
            critics=critics,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rates=critic_learning_rates,
            gamma=cfg.gamma,
            ent_coef=cfg.ent_coef,
            n_steps=cfg.n_steps,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=cfg.get("save_to_zoo", False),
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=cfg.get("ignore_trunc", True)
            )
        return agents

    @classmethod
    def from_tables(
            cls,
            actors={},
            critics={},
            action_spaces={},
            obs_spaces={},
            joint_space=None,
            actor_learning_rates={},
            critic_learning_rates={},
            gamma=1.0,
            ent_coef=0.0,
            clip_grad={},
            n_steps=5,
            exp_buffer_size=0,
            exp_buffer_replacement_prob=1.0,
            save_to_zoo=False,
            freeze_critic=0,
            freeze_actor=0,
            n_train_envs=1,
            ignore_trunc=True,
            ):
        ids = [agent_id for agent_id in actors.keys()]
        agent = cls(
            ids=ids,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rates=critic_learning_rates,
            gamma=gamma,
            ent_coef=ent_coef,
            clip_grad=clip_grad,
            n_steps=n_steps,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=save_to_zoo,
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=ignore_trunc,
            )

        # update the networks
        params = []
        for agent_id in ids:
            agent.actor_tabs[agent_id] = copy(actors[agent_id])
            params.append({
                "params": agent.actor_tabs[agent_id],
                "lr": actor_learning_rates[agent_id],
                })
            agent.critic_tabs[agent_id] = copy(critics[agent_id])
            params.append({
                "params": agent.critic_tabs[agent_id],
                "lr": critic_learning_rates[agent_id],
                })
        agent.optim = Adam(params,
                           lr=1e-4,
                           eps=1e-5)
        agent.saveables.update({
            "critic": agent.critic_tabs,
            "actor": agent.actor_tabs,
            "optim": agent.optim,
            })
        return agent


class TabularMAA2C(MultiAgent):
    def __init__(
            self,
            ids=[],
            action_spaces={},
            obs_spaces={},
            joint_space=None,
            actor_learning_rates={},
            critic_learning_rate=None,
            gamma=1.0,
            ent_coef=0.0,
            clip_grad={},
            n_steps=5,
            exp_buffer_size=0,
            exp_buffer_replacement_prob=1.0,
            save_to_zoo=False,
            freeze_critic=0,
            freeze_actor=0,
            n_train_envs=1,
            ignore_trunc=True,
        ):
        super().__init__(
            ids,
            action_spaces,
            obs_spaces,
            joint_space
            )
        state_size = list(obs_spaces.values())[0].shape[0]
        action_size = list(action_spaces.values())[0].n
        self.action_size = action_size

        params = []

        # Initialise the decentralised actors
        self.actor_tabs = {}
        self.actor_learning_rates = {}
        for agent_id in self.ids:
            self.actor_tabs[agent_id] = torch.zeros(state_size, action_size, requires_grad=True)
            self.actor_learning_rates[agent_id] = actor_learning_rates[agent_id]
            params.append({
                "params": self.actor_tabs[agent_id],
                "lr": self.actor_learning_rates[agent_id],
                })

        # Initialise the multi-agent critic
        self.critic_tab = torch.zeros(state_size, requires_grad=True)
        self.critic_learning_rate = critic_learning_rate
        params.append({
            "params": self.critic_tab,
            "lr": self.critic_learning_rate,
            })

        self.optim = Adam(params,
                          lr=params[0]["lr"],
                          eps=1e-5)

        # Initialise hyperparameters
        self.gamma = gamma
        self.gae_lambda = 1
        self.ent_coef = ent_coef
        self.n_steps = n_steps
        self.ignore_trunc = ignore_trunc
        self.clip_grad = defaultdict(lambda: False) 
        self.clip_grad.update(clip_grad)
        self.n_train_envs = n_train_envs
        self.freeze_critic = freeze_critic/self.n_train_envs
        self.freeze_actor = freeze_actor/self.n_train_envs
        # divide by n_train_envs since each call to update has n_train_envs transitions

        # Initialise update counters
        self.c_update = 0
        self.p_update = self.n_steps
        self.total_critic_updates = 0
        self.total_actor_updates = 0

        # Initialise memory
        self.last_transition = None
        self.last_n_transitions = ReplayBuffer(n_steps)
        self.exp_buffer_size = exp_buffer_size
        self.exp_buffer_replacement_prob = exp_buffer_replacement_prob
        self.exp_buffer = []

        self.saveables.update({
            "critic": self.critic_tab,
            "actor": self.actor_tabs,
            "optim": self.optim,
            })
        self.save_to_zoo = save_to_zoo

    def policy(self, obs, explore=True, agent_id=None):
        with torch.no_grad():
            if agent_id is None:
                p = {agent_id: self.policy(obs[agent_id], explore=explore, agent_id=agent_id)
                     for agent_id in self.ids}
            else:
                obs_torch = torch.Tensor(obs[agent_id])
                logits = obs_torch.matmul(self.actor_tabs[agent_id])
                p = torch.distributions.categorical.Categorical(logits=logits)
        return p

    def act(self, obs, explore=True, agent_id=None):
        if agent_id is None:
            return {agent_id: self.act(obs, explore=explore, agent_id=agent_id)
                    for agent_id in self.ids}
        else:
            policy = self.policy(obs, explore=explore, agent_id=agent_id)
            return policy.sample().unsqueeze(1).numpy()

    def update(self):
        self.c_update += 1
        critic_frozen = self.c_update <= self.freeze_critic
        actor_frozen = self.c_update <= self.freeze_actor
        if (critic_frozen and actor_frozen) or self.c_update % self.p_update != 0:
            return {}
        # generate G, V, lp_chosen, entropy
        joint_obs, agent_obs, actions = self._stack_transitions()
        A, G, V = self._compute_gae(joint_obs)
        lp_chosens, entropies = self._compute_policy_qty(agent_obs, actions)
        # update
        self.optim.zero_grad()
        critic_loss, critic_metrics = self._update_critic(G, V, frozen=critic_frozen)
        actor_loss, actor_metrics = self._update_actor(A, lp_chosens, entropies, frozen=actor_frozen)
        total_loss = critic_loss + actor_loss
        total_loss.backward()
        if not critic_frozen:
            self.total_critic_updates += 1
            critic_metrics[f"critic_grad"] = self.critic_tab.grad.detach().norm().item()
        if not actor_frozen:
            self.total_actor_updates += 1
            for agent_id in self.ids:
                actor_metrics[agent_id][f"actor_grad"] = self.actor_tabs[agent_id].grad.detach().norm().item()
        self.optim.step()

        # return
        train_metrics = {
            "central_critic": {
                "nTD": A.mean().item(),
                **critic_metrics
                },
            # "imp_weights": imp_weights.mean().item(),
            **actor_metrics
            }
        return train_metrics

    def _compute_gae(self, obs):
        V = obs.matmul(self.critic_tab).unsqueeze(-1)
        with torch.no_grad():
            G = torch.zeros_like(V)
            A = torch.zeros_like(V)
            V_n = self.last_transition.n_obs.matmul(self.critic_tab).unsqueeze(-1)
            G_n = V_n.clone()
            A_n = 0
            for i, t in enumerate(reversed(self.last_n_transitions)):
                term = t.terminated.unsqueeze(1)
                trunc = t.truncated.unsqueeze(1)
                r = t.reward.unsqueeze(1)
                if self.ignore_trunc:
                    V_n = ~(term|trunc) * V_n
                    G_n = ~(term|trunc) * G_n
                else:
                    V_n = (~trunc * (V_n * ~term)
                           + trunc * (V_n))
                    G_n = (~trunc * (G_n * ~term)
                           + trunc * (G_n))
                d = r + self.gamma * V_n - V[-i-1, :, :]
                A[-i-1, :, :] = d + self.gamma * self.gae_lambda * A_n
                G[-i-1, :, :] = r + self.gamma * G_n
                A_n = A[-i-1, :, :]
                G_n = G[-i-1, :, :]
                V_n = V[-i-1, :, :]
        return A, G, V

    def _compute_policy_qty(self, obs, acts, agent_id=None):
        if agent_id is None:
            lp_chosens, entropies = zip(*(self._compute_policy_qty(obs, acts, agent_id)
                                          for agent_id in self.ids))
            lp_chosens = dict(zip(self.ids, lp_chosens))
            entropies = dict(zip(self.ids, entropies))
            return lp_chosens, entropies
        else:
            logits = obs[agent_id].matmul(self.actor_tabs[agent_id])
            lp = torch.nn.functional.log_softmax(logits, dim=2)
            lp_chosen = lp.gather(2, acts[agent_id])
            entropy = -lp.exp().mul(lp).sum(dim=2).unsqueeze(2)
            return lp_chosen, entropy

    def _update_critic(self, G, V, frozen=False):
        if frozen:
            with torch.no_grad():
                critic_loss = torch.nn.functional.mse_loss(G, V)
        else:
            critic_loss = torch.nn.functional.mse_loss(G, V)
            if self.clip_grad.get("critic"):
                torch.nn.utils.clip_grad_norm_(self.critic_tab, self.clip_grad["critic"])

        critic_metrics = {
            "critic_loss": critic_loss.item()
            }
        return critic_loss, critic_metrics

    def _update_actor(self, A, lp_chosen, entropy, weights=1.0, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A,
                                                                         lp_chosen,
                                                                         entropy,
                                                                         weights,
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            if frozen:
                with torch.no_grad():
                    actor_loss = - (lp_chosen[agent_id].mul(A)
                                    + self.ent_coef*entropy[agent_id]).mul(weights).mean()
            else:
                actor_loss = - (lp_chosen[agent_id].mul(A)
                                + self.ent_coef*entropy[agent_id]).mul(weights).mean()
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_tabs[agent_id], self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy[agent_id].mean().item(),
                }
            return actor_loss, actor_metrics

    def store_transition(self, transitions, agent_id=None):
        # I expect this to come in as a dict of individual agent Transitions
        obs = {agent_id: torch.FloatTensor(t.obs)
               for agent_id, t in transitions.items()}
        action = {agent_id: torch.LongTensor(t.action)
                  for agent_id, t in transitions.items()}
        # Assumes common-payoff
        reward = torch.FloatTensor(list(transitions.values())[0].reward)
        terminated = torch.BoolTensor(list(transitions.values())[0].terminated)
        truncated = torch.BoolTensor(list(transitions.values())[0].truncated)
        joint_obs = torch.FloatTensor(list(transitions.values())[0].obs)
        n_obs = torch.FloatTensor(list(transitions.values())[0].n_obs)
        transition = Transition(
            obs=obs,
            action=action,
            n_obs=n_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            joint_obs=joint_obs,
            )
        self.last_transition = transition
        self.last_n_transitions.push(transition)
        self.store_exp(joint_obs)

    def _stack_transitions(self):
        joint_obs = torch.stack(list(t.joint_obs for t in self.last_n_transitions))
        agent_obs = {agent_id: torch.stack(list(t.obs[agent_id] for t in self.last_n_transitions))
                     for agent_id in self.ids}
        actions = {agent_id: torch.stack(list(t.action[agent_id] for t in self.last_n_transitions))
                   for agent_id in self.ids}
        return joint_obs, agent_obs, actions

    def store_exp(self, obs):
        if self.exp_buffer_size == 0:
            pass
        elif len(self.exp_buffer) >= self.exp_buffer_size:
            if np.random.random() < self.exp_buffer_replacement_prob:
                idx = np.random.randint(self.exp_buffer_size)
                self.exp_buffer[idx] = obs
        else:
            self.exp_buffer.append(obs)

    def add_agent(
            self,
            agent_id,
            action_space: gym.Space,
            obs_space: gym.Space,
            actor_hidden_size,
            critic_hidden_size,
            actor_learning_rate,
            critic_learning_rate,
            gamma,
            ent_coef,
            n_steps,
            save_to_zoo=False
            ):
        ...

    @property
    def info(self):
        return dict(
            type="TabularMAA2C",
            train_steps=self.c_update*self.n_train_envs,
            actor_train_updates=self.total_actor_updates,
            critic_train_updates=self.total_critic_updates,
            actor_learning_rates=self.actor_learning_rates,
            critic_learning_rate=self.critic_learning_rate,
            gamma=self.gamma,
            ent_coef=self.ent_coef,
            clip_grad={"actor": self.clip_grad["actor"], "critic": self.clip_grad["critic"]},
            n_steps=self.n_steps,
            exp_buffer_size=self.exp_buffer_size,
            exp_buffer_replacement_prob=self.exp_buffer_replacement_prob,
            ignore_trunc=self.ignore_trunc
            )
    ...
    @classmethod
    def from_config(cls, cfg, env):
        agent_ids = [a["agent_id"] for a in cfg.agents]
        assert len(agent_ids) == cfg.n_agents, \
            f"len(agent_ids) = {len(agent_ids)} (expected {cfg.n_agents})"
        action_spaces = {
            agent_id: env.action_space
            for agent_id in agent_ids
            }
        obs_spaces = {
            agent_id: env.observation_space
            for agent_id in agent_ids
            }
        if hasattr(env, "joint_space"):
            joint_space = env.joint_space
        else:
            joint_space = env.observation_space
        actor_learning_rates = {
            agent_id: cfg.lr.actor
            for agent_id in agent_ids
            }
        critic_learning_rate = cfg.lr.critic
        gamma = cfg.get("gamma", 1.0)
        ent_coef = cfg.get("ent_coef", 0.0)
        n_steps = cfg.get("n_steps", 5)
        clip_grad = cfg.get("clip_grad", {})
        save_to_zoo = cfg.get("save_to_zoo", False)
        ignore_trunc = cfg.get("ignore_trunc", True)
        if "exp_buffer" in cfg.keys():
            exp_buffer_size = cfg.exp_buffer.get("size", 0)
            exp_buffer_replacement_prob = cfg.exp_buffer.get("replacement_prob", 0.0)
        else:
            exp_buffer_size = 0
            exp_buffer_replacement_prob = 1.0

        if "freeze" in cfg.keys():
            freeze_critic = cfg.freeze.get("critic", 0)
            freeze_actor = cfg.freeze.get("actor", 0)
        else:
            freeze_critic = 0
            freeze_actor = 0

        if hasattr(env, "num_vec_envs"):
            n_train_envs = env.num_vec_envs
        elif hasattr(env, "vec_envs"):
            n_train_envs = len(env.vec_envs)
        else:
            n_train_envs = 1

        return cls(
            ids=agent_ids,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            ent_coef=ent_coef,
            n_steps=n_steps,
            clip_grad=clip_grad,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=save_to_zoo,
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=ignore_trunc,
            )

    def zoo_save(self, zoo_path, agent_save_names, env_cfg, evals, seed):
        agent_save_name_mapping = dict(zip(self.ids, agent_save_names))
        concat_name = "_".join(agent_save_names)
        # configs
        # actors
        # inelegant way of changing the evals into floats (from numpy.floats)
        # required so omegaconf can read
        float_evals = {
                "team": {k: v.item() for k, v in evals["team"].items()},
                "individual": {k: {k: v.item() for k, v in v.items()}
                               for k, v in evals["individual"].items()},
                }
        info = self.info
        for agent_id in self.ids:
            agent_name = agent_save_name_mapping[agent_id]
            model_pathname = osp.normpath(
                osp.join(zoo_path, "outputs", "actors", f"{agent_name}.pt")
                )
            config_pathname = osp.normpath(
                osp.join(zoo_path, "configs", "actors", f"{agent_name}.yaml")
                )
            os.makedirs(osp.dirname(model_pathname), exist_ok=True)
            os.makedirs(osp.dirname(config_pathname), exist_ok=True)
            torch.save(self.actor_tabs[agent_id], model_pathname)
            partner_agents = [agent_save_name_mapping[other_agent_id]
                              for other_agent_id in self.ids
                              if other_agent_id != agent_id]
            model_dict = dict(
                internal_name=agent_id,
                train_steps=info["train_steps"],
                train_updates=info["actor_train_updates"],
                learning_rate=info["actor_learning_rates"][agent_id],
                gamma=self.gamma,
                ent_coef=self.ent_coef,
                clip_grad=self.clip_grad["actor"],
                n_steps=self.n_steps,
                ignore_trunc=self.ignore_trunc,
                )
            actor_dict = dict(
                name=agent_name,
                model=model_dict,
                train_env=env_cfg,
                eval={
                    "team": float_evals["team"],
                    "individual": float_evals["individual"][agent_id],
                    },
                critic=concat_name,
                experience=concat_name,
                partner_agents=partner_agents,
                path_to_model=model_pathname,
                seed=seed,
                )
            actor_dict = OmegaConf.create(actor_dict)
            with open(config_pathname, "w") as f:
                f.write(OmegaConf.to_yaml(actor_dict))

        # critic
        model_pathname = osp.normpath(
            osp.join(zoo_path, "outputs", "critics", f"{concat_name}.pt")
            )
        config_pathname = osp.normpath(
            osp.join(zoo_path, "configs", "critics", f"{concat_name}.yaml")
            )
        os.makedirs(osp.dirname(model_pathname), exist_ok=True)
        os.makedirs(osp.dirname(config_pathname), exist_ok=True)
        torch.save(self.critic_tab, model_pathname)
        model_dict = dict(
            train_steps=info["train_steps"],
            train_updates=info["critic_train_updates"],
            learning_rate=info["critic_learning_rate"],
            gamma=self.gamma,
            clip_grad=self.clip_grad["critic"],
            n_steps=self.n_steps,
            ignore_trunc=self.ignore_trunc,
            )
        critic_dict = dict(
            name=concat_name,
            model=model_dict,
            train_env=env_cfg,
            eval=float_evals["team"],
            actors=agent_save_names,
            experience=concat_name,
            path_to_model=model_pathname,
            seed=seed,
            )
        critic_dict = OmegaConf.create(critic_dict)
        with open(config_pathname, "w") as f:
            f.write(OmegaConf.to_yaml(critic_dict))
        # experience
        experience_pathname = osp.normpath(
            osp.join(zoo_path, "outputs", "experience", f"{concat_name}.pt")
            )
        config_pathname = osp.normpath(
            osp.join(zoo_path, "configs", "experience", f"{concat_name}.yaml")
            )
        os.makedirs(osp.dirname(experience_pathname), exist_ok=True)
        os.makedirs(osp.dirname(config_pathname), exist_ok=True)
        if len(self.exp_buffer) > 0:
            save_buffer = torch.concat(self.exp_buffer)
        else:
            save_buffer = []
        torch.save(save_buffer, experience_pathname)
        buffer_dict = dict(
            max_size=self.exp_buffer_size,
            replacement_prob=self.exp_buffer_replacement_prob,
            size=len(self.exp_buffer),
            samples=len(save_buffer),
                )
        experience_dict = dict(
            name=concat_name,
            train_env=env_cfg,
            eval=float_evals["team"],
            experience=buffer_dict,
            critic=concat_name,
            actors=agent_save_names,
            path_to_experience=experience_pathname,
            seed=seed,
            )
        experience_dict = OmegaConf.create(experience_dict)
        with open(config_pathname, "w") as f:
            f.write(OmegaConf.to_yaml(experience_dict))
        return agent_save_name_mapping

    @classmethod
    def load_from_zoo(cls, cfg, name_id_mapping, env, zoo_path, reinit_critic=False):
        actors = {}
        agent_ids = list(name_id_mapping.values())
        critics = []
        exp_buffers = []
        for actor_name, agent_id in name_id_mapping.items():
            # load config
            path = osp.normpath(
                osp.join(zoo_path, "configs", "actors", f"{actor_name}.yaml")
                )
            actor_cfg = OmegaConf.load(path)
            actors[agent_id] = torch.load(actor_cfg.path_to_model)
            # load critic network
            critic_path = osp.normpath(
                osp.join(zoo_path, "configs", "critics", f"{actor_cfg.critic}.yaml")
                )
            critic_cfg = OmegaConf.load(critic_path)
            critics.append(torch.load(critic_cfg.path_to_model))
            # load experience
            exp_path = osp.normpath(
                osp.join(zoo_path, "configs", "experience", f"{actor_cfg.experience}.yaml")
                )
            exp_cfg = OmegaConf.load(exp_path)
            exp_buffers.append(torch.load(exp_cfg.path_to_experience))

        # Load env
        action_spaces = {
            agent_id: env.action_space
            for agent_id in agent_ids
            }
        obs_spaces = {
            agent_id: env.observation_space
            for agent_id in agent_ids
            }
        if hasattr(env, "joint_space"):
            joint_space = env.joint_space
        else:
            joint_space = env.observation_space

        critic = critics[0]  # Use the first critic

        actor_learning_rates = {
            agent_id: cfg.lr.actor
            for agent_id in agent_ids
            }
        critic_learning_rate = cfg.lr.critic
        if "exp_buffer" in cfg.keys():
            exp_buffer_size = cfg.exp_buffer.get("size", 0)
            exp_buffer_replacement_prob = cfg.exp_buffer.get("replacement_prob", 0.0)
        else:
            exp_buffer_size = 0
            exp_buffer_replacement_prob = 1.0
        if "freeze" in cfg.keys():
            freeze_critic = cfg.freeze.get("critic", 0)
            freeze_actor = cfg.freeze.get("actor", 0)
        else:
            freeze_critic = 0
            freeze_actor = 0

        if hasattr(env, "num_vec_envs"):
            n_train_envs = env.num_vec_envs
        elif hasattr(env, "vec_envs"):
            n_train_envs = len(env.vec_envs)
        else:
            n_train_envs = 1

        agents = TabularMAA2C.from_tables(
            actors=actors,
            critic=critic,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rate=critic_learning_rate,
            gamma=cfg.gamma,
            ent_coef=cfg.ent_coef,
            n_steps=cfg.n_steps,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=cfg.get("save_to_zoo", False),
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=cfg.get("ignore_trunc", True)
            )
        return agents

    @classmethod
    def from_tables(
            cls,
            actors={},
            critic=None,
            action_spaces={},
            obs_spaces={},
            joint_space=None,
            actor_learning_rates={},
            critic_learning_rate=None,
            gamma=1.0,
            ent_coef=0.0,
            clip_grad={},
            n_steps=5,
            exp_buffer_size=0,
            exp_buffer_replacement_prob=1.0,
            save_to_zoo=False,
            freeze_critic=0,
            freeze_actor=0,
            n_train_envs=1,
            ignore_trunc=True,
            ):
        ids = [agent_id for agent_id in actors.keys()]
        agent = cls(
            ids=ids,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            ent_coef=ent_coef,
            clip_grad=clip_grad,
            n_steps=n_steps,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=save_to_zoo,
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=ignore_trunc,
            )

        # update the networks
        params = []
        for agent_id in ids:
            agent.actor_tabs[agent_id] = copy(actors[agent_id])
            params.append({
                "params": agent.actor_tabs[agent_id],
                "lr": actor_learning_rates[agent_id],
                })
            agent.critic_tab = copy(critic)
            params.append({
                "params": agent.critic_tab,
                "lr": critic_learning_rate,
                })
        agent.optim = Adam(params,
                           lr=1e-4,
                           eps=1e-5)
        agent.saveables.update({
            "critic": agent.critic_tab,
            "actor": agent.actor_tabs,
            "optim": agent.optim,
            })
        return agent



class TabularIPPO(MultiAgent):
    def __init__(
            self,
            ids=[],
            action_spaces={},
            obs_spaces={},
            joint_space=None,
            actor_learning_rates={},
            critic_learning_rates={},
            gamma=1.0,
            gae_lambda=1.0,
            n_epochs=1,
            clip_coef=0.1,
            ent_coef=0.0,
            clip_grad={},
            n_steps=5,
            exp_buffer_size=0,
            exp_buffer_replacement_prob=1.0,
            save_to_zoo=False,
            freeze_critic=0,
            freeze_actor=0,
            n_train_envs=1,
            ignore_trunc=True,
        ):
        super().__init__(
            ids,
            action_spaces,
            obs_spaces,
            joint_space
            )
        state_size = list(obs_spaces.values())[0].shape[0]
        action_size = list(action_spaces.values())[0].n
        self.action_size = action_size

        params = []

        # Initialise the decentralised actors & critics
        self.actor_tabs = {}
        self.actor_learning_rates = {}
        self.critic_tabs = {}
        self.critic_learning_rates = {}
        for agent_id in self.ids:
            self.actor_tabs[agent_id] = torch.zeros(state_size, action_size, requires_grad=True)
            self.actor_learning_rates[agent_id] = actor_learning_rates[agent_id]
            self.critic_tabs[agent_id] = torch.zeros(state_size, requires_grad=True)
            self.critic_learning_rates[agent_id] = critic_learning_rates[agent_id]
            params.append({
                "params": self.actor_tabs[agent_id],
                "lr": self.actor_learning_rates[agent_id],
                })
            params.append({
                "params": self.critic_tabs[agent_id],
                "lr": self.critic_learning_rates[agent_id],
                })
        self.optim = Adam(params,
                          lr=params[0]["lr"],
                          eps=1e-5)

        # Initialise hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.n_steps = n_steps
        self.ignore_trunc = ignore_trunc
        self.clip_grad = defaultdict(lambda: False) 
        self.clip_grad.update(clip_grad)
        self.n_train_envs = n_train_envs
        self.freeze_critic = freeze_critic/self.n_train_envs
        self.freeze_actor = freeze_actor/self.n_train_envs
        # divide by n_train_envs since each call to update has n_train_envs transitions

        # Initialise update counters
        self.c_update = 0
        self.p_update = self.n_steps
        self.total_critic_updates = 0
        self.total_actor_updates = 0

        # Initialise memory
        self.last_transition = None
        self.last_n_transitions = ReplayBuffer(n_steps)
        self.exp_buffer_size = exp_buffer_size
        self.exp_buffer_replacement_prob = exp_buffer_replacement_prob
        self.exp_buffer = {agent_id: [] for agent_id in self.ids}

        self.saveables.update({
            "critic": self.critic_tabs,
            "actor": self.actor_tabs,
            "optim": self.optim,
            })
        self.save_to_zoo = save_to_zoo

    def policy(self, obs, explore=True, agent_id=None):
        with torch.no_grad():
            if agent_id is None:
                p = {agent_id: self.policy(obs, explore=explore, agent_id=agent_id)
                     for agent_id in self.ids}
            else:
                obs_torch = torch.Tensor(obs[agent_id])
                logits = obs_torch.matmul(self.actor_tabs[agent_id])
                p = torch.distributions.categorical.Categorical(logits=logits)
        return p

    def act(self, obs, explore=True, agent_id=None):
        if agent_id is None:
            return {agent_id: self.act(obs, explore=explore, agent_id=agent_id)
                    for agent_id in self.ids}
        else:
            policy = self.policy(obs, explore=explore, agent_id=agent_id)
            return policy.sample().unsqueeze(1).numpy()

    def update(self):
        self.c_update += 1
        critic_frozen = self.c_update <= self.freeze_critic
        actor_frozen = self.c_update <= self.freeze_actor
        if (critic_frozen and actor_frozen) or self.c_update % self.p_update != 0:
            return {}
        # generate G, V, lp_chosen, entropy
        _, agent_obs, actions = self._stack_transitions()
        As, Gs, Vs = self._compute_gae(agent_obs)
        lp_chosens, entropies = self._compute_policy_qty(agent_obs, actions)
        # update
        with torch.no_grad():
            orig_lp_chosens, _ = self._compute_policy_qty(agent_obs, actions)
        # update loop:
        for epoch in range(self.n_epochs):
            self.optim.zero_grad()
            new_lp_chosens, new_entropies = self._compute_policy_qty(agent_obs, actions)
            policy_ratio = {agent_id: torch.exp(new_lp_chosens[agent_id] - orig_lp_chosens[agent_id])
                            for agent_id in self.ids}
            new_Vs = {agent_id: agent_obs[agent_id].matmul(self.critic_tabs[agent_id]).unsqueeze(-1)
                      for agent_id in self.ids}
            critic_loss, critic_metrics = self._update_critic(Gs, new_Vs, frozen=critic_frozen)
            actor_loss, actor_metrics = self._update_actor(As, policy_ratio, new_entropies, frozen=actor_frozen)
            total_loss = critic_loss + actor_loss
            total_loss.backward()
            self.optim.step()
            if not critic_frozen:
                self.total_critic_updates += 1
                for agent_id in self.ids:
                    critic_metrics[agent_id]["critic_grad"] = self.critic_tabs[agent_id].grad.detach().norm().item()
            if not actor_frozen:
                self.total_actor_updates += 1
                for agent_id in self.ids:
                    actor_metrics[agent_id]["actor_grad"] = self.actor_tabs[agent_id].grad.detach().norm().item()

        # return
        train_metrics = {
            **critic_metrics,
            **actor_metrics
            }
        return train_metrics

    def _compute_gae(self, obs, agent_id=None):
        if agent_id is None:
            As, Gs, Vs = zip(*(self._compute_gae(obs, agent_id=agent_id)
                           for agent_id in self.ids))
            As = dict(zip(self.ids, As))
            Gs = dict(zip(self.ids, Gs))
            Vs = dict(zip(self.ids, Vs))
            return As, Gs, Vs
        else:
            V = obs[agent_id].matmul(self.critic_tabs[agent_id]).unsqueeze(-1)
            with torch.no_grad():
                G = torch.zeros_like(V)
                A = torch.zeros_like(V)
                V_n = self.last_transition.n_obs[agent_id].matmul(self.critic_tabs[agent_id]).unsqueeze(-1)
                G_n = V_n.clone()
                A_n = 0
                for i, t in enumerate(reversed(self.last_n_transitions)):
                    term = t.terminated.unsqueeze(1)
                    trunc = t.truncated.unsqueeze(1)
                    r = t.reward.unsqueeze(1)
                    if self.ignore_trunc:
                        V_n = ~(term|trunc) * V_n
                        G_n = ~(term|trunc) * G_n
                    else:
                        V_n = (~trunc * (V_n * ~term)
                               + trunc * (V_n))
                        G_n = (~trunc * (G_n * ~term)
                               + trunc * (G_n))
                    d = r + self.gamma * V_n - V[-i-1, :, :]
                    A[-i-1, :, :] = d + self.gamma * self.gae_lambda * A_n
                    G[-i-1, :, :] = r + self.gamma * G_n
                    A_n = A[-i-1, :, :]
                    G_n = G[-i-1, :, :]
                    V_n = V[-i-1, :, :]
            return A, G, V

    def _compute_policy_qty(self, obs, acts, agent_id=None):
        if agent_id is None:
            lp_chosens, entropies = zip(*(self._compute_policy_qty(obs, acts, agent_id)
                                          for agent_id in self.ids))
            lp_chosens = dict(zip(self.ids, lp_chosens))
            entropies = dict(zip(self.ids, entropies))
            return lp_chosens, entropies
        else:
            logits = obs[agent_id].matmul(self.actor_tabs[agent_id])
            lp = torch.nn.functional.log_softmax(logits, dim=2)
            lp_chosen = lp.gather(2, acts[agent_id])
            entropy = -lp.exp().mul(lp).sum(dim=2).unsqueeze(2)
            return lp_chosen, entropy

    def _update_critic(self, G, V, frozen=False, agent_id=None):
        if agent_id is None:
            total_loss = 0
            critic_metrics = {}
            for agent_id in self.ids:
                critic_loss, critic_metrics[agent_id] = self._update_critic(G[agent_id], V[agent_id],
                                                                            agent_id=agent_id,
                                                                            frozen=frozen)
                total_loss += critic_loss
            return total_loss, critic_metrics
        else:
            if frozen:
                with torch.no_grad():
                    critic_loss = torch.nn.functional.mse_loss(G, V)
            else:
                critic_loss = torch.nn.functional.mse_loss(G, V)
                if self.clip_grad.get("critic"):
                    torch.nn.utils.clip_grad_norm_(self.critic_tabs[agent_id], self.clip_grad["critic"])
            critic_metrics = {
                "critic_loss": critic_loss.item()
                }
            return critic_loss, critic_metrics

    def _update_actor(self, A, ratio, entropy, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A[agent_id],
                                                                         ratio[agent_id],
                                                                         entropy[agent_id],
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            A_norm = (A-A.mean())/(A.std() + 1e-8)
            if frozen:
                with torch.no_grad():
                    policy_loss = torch.min(
                        A_norm * ratio,
                        A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                        ).mean()
                    actor_loss = -(policy_loss + self.ent_coef*entropy.mean())
            else:
                policy_loss = torch.min(
                    A_norm * ratio,
                    A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                    ).mean()
                actor_loss = -(policy_loss + self.ent_coef*entropy.mean())
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_tabs[agent_id], self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy.mean().item(),
                }
            return actor_loss, actor_metrics

    def store_transition(self, transitions, agent_id=None):
        # I expect this to come in as a dict of individual agent Transitions
        obs = {agent_id: torch.FloatTensor(t.obs)
               for agent_id, t in transitions.items()}
        n_obs = {agent_id: torch.FloatTensor(t.n_obs)
               for agent_id, t in transitions.items()}
        action = {agent_id: torch.LongTensor(t.action)
                  for agent_id, t in transitions.items()}
        # Assumes common-payoff
        reward = torch.FloatTensor(list(transitions.values())[0].reward)
        terminated = torch.BoolTensor(list(transitions.values())[0].terminated)
        truncated = torch.BoolTensor(list(transitions.values())[0].truncated)
        joint_obs = torch.FloatTensor(list(transitions.values())[0].obs)
        transition = Transition(
            obs=obs,
            action=action,
            n_obs=n_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            joint_obs=joint_obs,
            )
        self.last_transition = transition
        self.last_n_transitions.push(transition)
        self.store_exp(obs)

    def _stack_transitions(self):
        joint_obs = torch.stack(list(t.joint_obs for t in self.last_n_transitions))
        agent_obs = {agent_id: torch.stack(list(t.obs[agent_id] for t in self.last_n_transitions))
                     for agent_id in self.ids}
        actions = {agent_id: torch.stack(list(t.action[agent_id] for t in self.last_n_transitions))
                   for agent_id in self.ids}
        return joint_obs, agent_obs, actions

    def store_exp(self, obs, agent_id=None):
        if agent_id is None:
            for agent_id in self.ids:
                self.store_exp(obs, agent_id=agent_id)
        else:
            if self.exp_buffer_size == 0:
                pass
            elif len(self.exp_buffer[agent_id]) >= self.exp_buffer_size:
                if np.random.random() < self.exp_buffer_replacement_prob:
                    idx = np.random.randint(self.exp_buffer_size)
                    self.exp_buffer[idx] = obs[agent_id]
            else:
                self.exp_buffer[agent_id].append(obs[agent_id])

    def add_agent(
            self,
            agent_id,
            action_space: gym.Space,
            obs_space: gym.Space,
            actor_hidden_size,
            critic_hidden_size,
            actor_learning_rate,
            critic_learning_rate,
            gamma,
            ent_coef,
            n_steps,
            save_to_zoo=False
            ):
        ...

    @property
    def info(self):
        return dict(
            type="TabularIPPO",
            train_steps=self.c_update*self.n_train_envs,
            actor_train_updates=self.total_actor_updates,
            critic_train_updates=self.total_critic_updates,
            actor_learning_rates=self.actor_learning_rates,
            critic_learning_rates=self.critic_learning_rates,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_epochs=self.n_epochs,
            clip_coef=self.clip_coef,
            ent_coef=self.ent_coef,
            clip_grad={"actor": self.clip_grad["actor"], "critic": self.clip_grad["critic"]},
            n_steps=self.n_steps,
            exp_buffer_size=self.exp_buffer_size,
            exp_buffer_replacement_prob=self.exp_buffer_replacement_prob,
            ignore_trunc=self.ignore_trunc
            )
    ...
    @classmethod
    def from_config(cls, cfg, env):
        agent_ids = [a["agent_id"] for a in cfg.agents]
        assert len(agent_ids) == cfg.n_agents, \
            f"len(agent_ids) = {len(agent_ids)} (expected {cfg.n_agents})"
        action_spaces = {
            agent_id: env.action_space
            for agent_id in agent_ids
            }
        obs_spaces = {
            agent_id: env.observation_space
            for agent_id in agent_ids
            }
        if hasattr(env, "joint_space"):
            joint_space = env.joint_space
        else:
            joint_space = env.observation_space
        actor_learning_rates = {
            agent_id: cfg.lr.actor
            for agent_id in agent_ids
            }
        critic_learning_rates = {
            agent_id: cfg.lr.critic
            for agent_id in agent_ids
            }
        gamma = cfg.get("gamma", 1.0)
        ent_coef = cfg.get("ent_coef", 0.0)
        n_steps = cfg.get("n_steps", 5)
        clip_grad = cfg.get("clip_grad", {})
        save_to_zoo = cfg.get("save_to_zoo", False)
        ignore_trunc = cfg.get("ignore_trunc", True)
        if "exp_buffer" in cfg.keys():
            exp_buffer_size = cfg.exp_buffer.get("size", 0)
            exp_buffer_replacement_prob = cfg.exp_buffer.get("replacement_prob", 0.0)
        else:
            exp_buffer_size = 0
            exp_buffer_replacement_prob = 1.0

        if "freeze" in cfg.keys():
            freeze_critic = cfg.freeze.get("critic", 0)
            freeze_actor = cfg.freeze.get("actor", 0)
        else:
            freeze_critic = 0
            freeze_actor = 0

        if hasattr(env, "num_vec_envs"):
            n_train_envs = env.num_vec_envs
        elif hasattr(env, "vec_envs"):
            n_train_envs = len(env.vec_envs)
        else:
            n_train_envs = 1

        return cls(
            ids=agent_ids,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rates=critic_learning_rates,
            gamma=gamma,
            ent_coef=ent_coef,
            n_steps=n_steps,
            clip_grad=clip_grad,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=save_to_zoo,
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=ignore_trunc,
            )

    def zoo_save(self, zoo_path, agent_save_names, env_cfg, evals, seed):
        agent_save_name_mapping = dict(zip(self.ids, agent_save_names))
        concat_name = "_".join(agent_save_names)
        # configs
        # actors
        # inelegant way of changing the evals into floats (from numpy.floats)
        # required so omegaconf can read
        float_evals = {
                "team": {k: v.item() for k, v in evals["team"].items()},
                "individual": {k: {k: v.item() for k, v in v.items()}
                               for k, v in evals["individual"].items()},
                }
        info = self.info
        for agent_id in self.ids:
            agent_name = agent_save_name_mapping[agent_id]
            partner_agents = [agent_save_name_mapping[other_agent_id]
                              for other_agent_id in self.ids
                              if other_agent_id != agent_id]
            # Actor
            actor_model_pathname = osp.normpath(
                osp.join(zoo_path, "outputs", "actors", f"{agent_name}.pt")
                )
            actor_config_pathname = osp.normpath(
                osp.join(zoo_path, "configs", "actors", f"{agent_name}.yaml")
                )
            os.makedirs(osp.dirname(actor_model_pathname), exist_ok=True)
            os.makedirs(osp.dirname(actor_config_pathname), exist_ok=True)
            torch.save(self.actor_tabs[agent_id], actor_model_pathname)
            actor_model_dict = dict(
                internal_name=agent_id,
                train_steps=info["train_steps"],
                train_updates=info["actor_train_updates"],
                learning_rate=info["actor_learning_rates"][agent_id],
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_epochs=self.n_epochs,
                clip_coef=self.clip_coef,
                ent_coef=self.ent_coef,
                clip_grad=self.clip_grad["actor"],
                n_steps=self.n_steps,
                ignore_trunc=self.ignore_trunc,
                )
            actor_dict = dict(
                name=agent_name,
                model=actor_model_dict,
                train_env=env_cfg,
                eval={
                    "team": float_evals["team"],
                    "individual": float_evals["individual"][agent_id],
                    },
                critic=agent_name,
                experience=agent_name,
                partner_agents=partner_agents,
                path_to_model=actor_model_pathname,
                seed=seed,
                )
            actor_dict = OmegaConf.create(actor_dict)
            with open(actor_config_pathname, "w") as f:
                f.write(OmegaConf.to_yaml(actor_dict))

            # Critic
            critic_model_pathname = osp.normpath(
                osp.join(zoo_path, "outputs", "critics", f"{agent_name}.pt")
                )
            critic_config_pathname = osp.normpath(
                osp.join(zoo_path, "configs", "critics", f"{agent_name}.yaml")
                )
            os.makedirs(osp.dirname(critic_model_pathname), exist_ok=True)
            os.makedirs(osp.dirname(critic_config_pathname), exist_ok=True)
            torch.save(self.critic_tabs[agent_id], critic_model_pathname)
            critic_model_dict = dict(
                internal_name=agent_id,
                train_steps=info["train_steps"],
                train_updates=info["critic_train_updates"],
                learning_rate=info["critic_learning_rates"][agent_id],
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_epochs=self.n_epochs,
                clip_coef=self.clip_coef,
                ent_coef=self.ent_coef,
                clip_grad=self.clip_grad["critic"],
                n_steps=self.n_steps,
                ignore_trunc=self.ignore_trunc,
                )
            critic_dict = dict(
                name=agent_name,
                model=critic_model_dict,
                train_env=env_cfg,
                eval={
                    "team": float_evals["team"],
                    "individual": float_evals["individual"][agent_id],
                    },
                actor=agent_name,
                experience=agent_name,
                partner_agents=partner_agents,
                path_to_model=critic_model_pathname,
                seed=seed,
                )
            critic_dict = OmegaConf.create(critic_dict)
            with open(critic_config_pathname, "w") as f:
                f.write(OmegaConf.to_yaml(critic_dict))

            # experience
            experience_pathname = osp.normpath(
                osp.join(zoo_path, "outputs", "experience", f"{agent_name}.pt")
                )
            config_pathname = osp.normpath(
                osp.join(zoo_path, "configs", "experience", f"{agent_name}.yaml")
                )
            os.makedirs(osp.dirname(experience_pathname), exist_ok=True)
            os.makedirs(osp.dirname(config_pathname), exist_ok=True)
            if len(self.exp_buffer) > 0:
                save_buffer = torch.concat(self.exp_buffer[agent_id])
            else:
                save_buffer = []
            torch.save(save_buffer, experience_pathname)
            buffer_dict = dict(
                max_size=self.exp_buffer_size,
                replacement_prob=self.exp_buffer_replacement_prob,
                size=len(self.exp_buffer),
                samples=len(save_buffer),
                    )
            experience_dict = dict(
                name=agent_name,
                train_env=env_cfg,
                eval=float_evals["team"],
                experience=buffer_dict,
                critics=agent_save_names,
                actors=agent_save_names,
                path_to_experience=experience_pathname,
                seed=seed,
                )
            experience_dict = OmegaConf.create(experience_dict)
            with open(config_pathname, "w") as f:
                f.write(OmegaConf.to_yaml(experience_dict))
        return agent_save_name_mapping

    @classmethod
    def load_from_zoo(cls, cfg, name_id_mapping, env, zoo_path):
        actors = {}
        critics = {}
        agent_ids = list(name_id_mapping.values())
        exp_buffers = []
        for actor_name, agent_id in name_id_mapping.items():
            # load config
            actor_cfg_path = osp.normpath(
                osp.join(zoo_path, "configs", "actors", f"{actor_name}.yaml")
                )
            actor_cfg = OmegaConf.load(actor_cfg_path)
            actors[agent_id] = torch.load(actor_cfg.path_to_model)
            # load critic network
            critic_cfg_path = osp.normpath(
                osp.join(zoo_path, "configs", "critics", f"{actor_cfg.critic}.yaml")
                )
            critic_cfg = OmegaConf.load(critic_cfg_path)
            critics[agent_id] = torch.load(critic_cfg.path_to_model)
            # load experience
            exp_path = osp.normpath(
                osp.join(zoo_path, "configs", "experience", f"{actor_cfg.experience}.yaml")
                )
            exp_cfg = OmegaConf.load(exp_path)
            exp_buffers.append(torch.load(exp_cfg.path_to_experience))

        # Load env
        action_spaces = {
            agent_id: env.action_space
            for agent_id in agent_ids
            }
        obs_spaces = {
            agent_id: env.observation_space
            for agent_id in agent_ids
            }
        if hasattr(env, "joint_space"):
            joint_space = env.joint_space
        else:
            joint_space = env.observation_space

        actor_learning_rates = {
            agent_id: cfg.lr.actor
            for agent_id in agent_ids
            }
        critic_learning_rates = {
            agent_id: cfg.lr.critic
            for agent_id in agent_ids
            }
        if "exp_buffer" in cfg.keys():
            exp_buffer_size = cfg.exp_buffer.get("size", 0)
            exp_buffer_replacement_prob = cfg.exp_buffer.get("replacement_prob", 0.0)
        else:
            exp_buffer_size = 0
            exp_buffer_replacement_prob = 1.0
        if "freeze" in cfg.keys():
            freeze_critic = cfg.freeze.get("critic", 0)
            freeze_actor = cfg.freeze.get("actor", 0)
        else:
            freeze_critic = 0
            freeze_actor = 0

        if hasattr(env, "num_vec_envs"):
            n_train_envs = env.num_vec_envs
        elif hasattr(env, "vec_envs"):
            n_train_envs = len(env.vec_envs)
        else:
            n_train_envs = 1

        agents = cls.from_tables(
            actors=actors,
            critics=critics,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rates=critic_learning_rates,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            n_epochs=cfg.n_epochs,
            clip_coef=cfg.clip_coef,
            ent_coef=cfg.ent_coef,
            n_steps=cfg.n_steps,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=cfg.get("save_to_zoo", False),
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=cfg.get("ignore_trunc", True)
            )
        return agents

    @classmethod
    def from_tables(
            cls,
            actors={},
            critics={},
            action_spaces={},
            obs_spaces={},
            joint_space=None,
            actor_learning_rates={},
            critic_learning_rates={},
            gamma=1.0,
            gae_lambda=1.0,
            n_epochs=1,
            clip_coef=0.1,
            ent_coef=0.0,
            clip_grad={},
            n_steps=5,
            exp_buffer_size=0,
            exp_buffer_replacement_prob=1.0,
            save_to_zoo=False,
            freeze_critic=0,
            freeze_actor=0,
            n_train_envs=1,
            ignore_trunc=True,
            ):
        ids = [agent_id for agent_id in actors.keys()]
        agent = cls(
            ids=ids,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rates=critic_learning_rates,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_epochs=n_epochs,
            clip_coef=clip_coef,
            ent_coef=ent_coef,
            clip_grad=clip_grad,
            n_steps=n_steps,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=save_to_zoo,
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=ignore_trunc,
            )

        # update the networks
        params = []
        for agent_id in ids:
            agent.actor_tabs[agent_id] = copy(actors[agent_id])
            params.append({
                "params": agent.actor_tabs[agent_id],
                "lr": actor_learning_rates[agent_id],
                })
            agent.critic_tabs[agent_id] = copy(critics[agent_id])
            params.append({
                "params": agent.critic_tabs[agent_id],
                "lr": critic_learning_rates[agent_id],
                })
        agent.optim = Adam(params,
                           lr=1e-4,
                           eps=1e-5)
        agent.saveables.update({
            "critic": agent.critic_tabs,
            "actor": agent.actor_tabs,
            "optim": agent.optim,
            })
        return agent

class TabularDeIPPO(TabularIPPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self):
        self.c_update += 1
        critic_frozen = self.c_update <= self.freeze_critic
        actor_frozen = self.c_update <= self.freeze_actor
        if (critic_frozen and actor_frozen) or self.c_update % self.p_update != 0:
            return {}
        # generate G, V, lp_chosen, entropy
        _, agent_obs, actions = self._stack_transitions()
        As, Gs, Vs = self._compute_gae(agent_obs)
        lp_chosens, entropies = self._compute_policy_qty(agent_obs, actions)
        # update
        with torch.no_grad():
            behav_lp_chosens, _ = self._compute_policy_qty(agent_obs, actions)
            proxi_lp_chosens = behav_lp_chosens
        # update loop:
        for epoch in range(self.n_epochs):
            self.optim.zero_grad()
            new_lp_chosens, new_entropies = self._compute_policy_qty(agent_obs, actions)
            policy_ratio = {agent_id: torch.exp(new_lp_chosens[agent_id] - proxi_lp_chosens[agent_id])
                            for agent_id in self.ids}
            imp_weight = {agent_id: torch.exp(proxi_lp_chosens[agent_id] - behav_lp_chosens[agent_id])
                          for agent_id in self.ids}
            new_Vs = {agent_id: agent_obs[agent_id].matmul(self.critic_tabs[agent_id]).unsqueeze(-1)
                      for agent_id in self.ids}
            critic_loss, critic_metrics = self._update_critic(Gs, new_Vs, imp_weight, frozen=critic_frozen)
            actor_loss, actor_metrics = self._update_actor(As, policy_ratio, new_entropies, imp_weight, frozen=actor_frozen)
            total_loss = critic_loss + actor_loss
            total_loss.backward()
            self.optim.step()
            if not critic_frozen:
                self.total_critic_updates += 1
                for agent_id in self.ids:
                    critic_metrics[agent_id]["critic_grad"] = self.critic_tabs[agent_id].grad.detach().norm().item()
            if not actor_frozen:
                self.total_actor_updates += 1
                for agent_id in self.ids:
                    actor_metrics[agent_id]["actor_grad"] = self.actor_tabs[agent_id].grad.detach().norm().item()

        # return
        train_metrics = {
            **critic_metrics,
            **actor_metrics
            }
        return train_metrics

    def _update_critic(self, G, V, imp_weight, frozen=False, agent_id=None):
        if agent_id is None:
            total_loss = 0
            critic_metrics = {}
            for agent_id in self.ids:
                critic_loss, critic_metrics[agent_id] = self._update_critic(G[agent_id], V[agent_id],
                                                                            imp_weight[agent_id],
                                                                            agent_id=agent_id,
                                                                            frozen=frozen)
                total_loss += critic_loss
            return total_loss, critic_metrics
        else:
            if frozen:
                with torch.no_grad():
                    critic_loss = (G-V).pow(2).mul(imp_weight).mean()
            else:
                critic_loss = (G-V).pow(2).mul(imp_weight).mean()
                if self.clip_grad.get("critic"):
                    torch.nn.utils.clip_grad_norm_(self.critic_tabs[agent_id], self.clip_grad["critic"])
            critic_metrics = {
                "critic_loss": critic_loss.item()
                }
            return critic_loss, critic_metrics

    def _update_actor(self, A, ratio, entropy, imp_weight, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A[agent_id],
                                                                         ratio[agent_id],
                                                                         entropy[agent_id],
                                                                         imp_weight[agent_id],
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            A_norm = (A-A.mean())/(A.std() + 1e-8)
            if frozen:
                with torch.no_grad():
                    policy_loss = imp_weight.mul(torch.min(
                        A_norm * ratio,
                        A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                        )).mean()
                    actor_loss = -(policy_loss + self.ent_coef*entropy.mean())
            else:
                policy_loss = imp_weight.mul(torch.min(
                    A_norm * ratio,
                    A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                    )).mean()
                actor_loss = -(policy_loss + self.ent_coef*entropy.mean())
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_tabs[agent_id], self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy.mean().item(),
                "imp_weight/max": imp_weight.max().item(),
                "imp_weight/min": imp_weight.min().item(),
                "imp_weight/mean": imp_weight.mean().item(),
                }
            return actor_loss, actor_metrics

    @property
    def info(self):
        return dict(
            type="TabularDeIPPO",
            train_steps=self.c_update*self.n_train_envs,
            actor_train_updates=self.total_actor_updates,
            critic_train_updates=self.total_critic_updates,
            actor_learning_rates=self.actor_learning_rates,
            critic_learning_rates=self.critic_learning_rates,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_epochs=self.n_epochs,
            clip_coef=self.clip_coef,
            ent_coef=self.ent_coef,
            clip_grad={"actor": self.clip_grad["actor"], "critic": self.clip_grad["critic"]},
            n_steps=self.n_steps,
            exp_buffer_size=self.exp_buffer_size,
            exp_buffer_replacement_prob=self.exp_buffer_replacement_prob,
            ignore_trunc=self.ignore_trunc
            )

class TabularMAPPO(MultiAgent):
    def __init__(
            self,
            ids=[],
            action_spaces={},
            obs_spaces={},
            joint_space=None,
            actor_learning_rates={},
            critic_learning_rate=None,
            gamma=1.0,
            gae_lambda=1.0,
            n_epochs=1,
            clip_coef=0.1,
            ent_coef=0.0,
            clip_grad={},
            n_steps=5,
            exp_buffer_size=0,
            exp_buffer_replacement_prob=1.0,
            save_to_zoo=False,
            freeze_critic=0,
            freeze_actor=0,
            n_train_envs=1,
            ignore_trunc=True,
        ):
        super().__init__(
            ids,
            action_spaces,
            obs_spaces,
            joint_space
            )
        state_size = list(obs_spaces.values())[0].shape[0]
        action_size = list(action_spaces.values())[0].n
        self.action_size = action_size

        params = []

        # Initialise the decentralised actors
        self.actor_tabs = {}
        self.actor_learning_rates = {}
        for agent_id in self.ids:
            self.actor_tabs[agent_id] = torch.zeros(state_size, action_size, requires_grad=True)
            self.actor_learning_rates[agent_id] = actor_learning_rates[agent_id]
            params.append({
                "params": self.actor_tabs[agent_id],
                "lr": self.actor_learning_rates[agent_id],
                })

        # Initialise the multi-agent critic
        self.critic_tab = torch.zeros(state_size, requires_grad=True)
        self.critic_learning_rate = critic_learning_rate
        params.append({
            "params": self.critic_tab,
            "lr": self.critic_learning_rate,
            })

        self.optim = Adam(params,
                          lr=params[0]["lr"],
                          eps=1e-5)

        # Initialise hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.n_steps = n_steps
        self.ignore_trunc = ignore_trunc
        self.clip_grad = defaultdict(lambda: False) 
        self.clip_grad.update(clip_grad)
        self.n_train_envs = n_train_envs
        self.freeze_critic = freeze_critic/self.n_train_envs
        self.freeze_actor = freeze_actor/self.n_train_envs
        # divide by n_train_envs since each call to update has n_train_envs transitions

        # Initialise update counters
        self.c_update = 0
        self.p_update = self.n_steps
        self.total_critic_updates = 0
        self.total_actor_updates = 0

        # Initialise memory
        self.last_transition = None
        self.last_n_transitions = ReplayBuffer(n_steps)
        self.exp_buffer_size = exp_buffer_size
        self.exp_buffer_replacement_prob = exp_buffer_replacement_prob
        self.exp_buffer = []

        self.saveables.update({
            "critic": self.critic_tab,
            "actor": self.actor_tabs,
            "optim": self.optim,
            })
        self.save_to_zoo = save_to_zoo

    def policy(self, obs, explore=True, agent_id=None):
        with torch.no_grad():
            if agent_id is None:
                p = {agent_id: self.policy(obs, explore=explore, agent_id=agent_id)
                     for agent_id in self.ids}
            else:
                obs_torch = torch.Tensor(obs[agent_id])
                logits = obs_torch.matmul(self.actor_tabs[agent_id])
                p = torch.distributions.categorical.Categorical(logits=logits)
        return p

    def act(self, obs, explore=True, agent_id=None):
        if agent_id is None:
            return {agent_id: self.act(obs, explore=explore, agent_id=agent_id)
                    for agent_id in self.ids}
        else:
            policy = self.policy(obs, explore=explore, agent_id=agent_id)
            return policy.sample().unsqueeze(1).numpy()

    def update(self):
        self.c_update += 1
        critic_frozen = self.c_update <= self.freeze_critic
        actor_frozen = self.c_update <= self.freeze_actor
        if (critic_frozen and actor_frozen) or self.c_update % self.p_update != 0:
            return {}
        # generate G, V, lp_chosen, entropy
        joint_obs, agent_obs, actions = self._stack_transitions()
        A, G, V = self._compute_gae(joint_obs)
        # update

        with torch.no_grad():
            orig_lp_chosens, _ = self._compute_policy_qty(agent_obs, actions)
        # update loop:
        for epoch in range(self.n_epochs):
            self.optim.zero_grad()
            new_lp_chosens, new_entropies = self._compute_policy_qty(agent_obs, actions)
            policy_ratio = {agent_id: torch.exp(new_lp_chosens[agent_id] - orig_lp_chosens[agent_id])
                            for agent_id in self.ids}
            new_V = joint_obs.matmul(self.critic_tab).unsqueeze(-1)
            critic_loss, critic_metrics = self._update_critic(G, new_V, frozen=critic_frozen)
            actor_loss, actor_metrics = self._update_actor(A, policy_ratio, new_entropies, frozen=actor_frozen)
            total_loss = critic_loss + actor_loss
            total_loss.backward()
            self.optim.step()
            if not critic_frozen:
                self.total_critic_updates += 1
                critic_metrics["critic_grad"] = self.critic_tab.grad.detach().norm().item()
            if not actor_frozen:
                self.total_actor_updates += 1
                for agent_id in self.ids:
                    actor_metrics[agent_id]["actor_grad"] = self.actor_tabs[agent_id].grad.detach().norm().item()
        # return
        train_metrics = {
            "central_critic": {
                "nTD": A.mean().item(),
                **critic_metrics
                },
            # "imp_weights": imp_weights.mean().item(),
            **actor_metrics
            }
        return train_metrics

    def _compute_gae(self, obs):
        V = obs.matmul(self.critic_tab).unsqueeze(-1)
        with torch.no_grad():
            G = torch.zeros_like(V)
            A = torch.zeros_like(V)
            V_n = self.last_transition.n_obs.matmul(self.critic_tab).unsqueeze(-1)
            G_n = V_n.clone()
            A_n = 0
            for i, t in enumerate(reversed(self.last_n_transitions)):
                term = t.terminated.unsqueeze(1)
                trunc = t.truncated.unsqueeze(1)
                r = t.reward.unsqueeze(1)
                if self.ignore_trunc:
                    V_n = ~(term|trunc) * V_n
                    G_n = ~(term|trunc) * G_n
                else:
                    V_n = (~trunc * (V_n * ~term)
                           + trunc * (V_n))
                    G_n = (~trunc * (G_n * ~term)
                           + trunc * (G_n))
                d = r + self.gamma * V_n - V[-i-1, :, :]
                A[-i-1, :, :] = d + self.gamma * self.gae_lambda * A_n
                G[-i-1, :, :] = r + self.gamma * G_n
                A_n = A[-i-1, :, :]
                G_n = G[-i-1, :, :]
                V_n = V[-i-1, :, :]
        return A, G, V

    def _compute_policy_qty(self, obs, acts, agent_id=None):
        if agent_id is None:
            lp_chosens, entropies = zip(*(self._compute_policy_qty(obs, acts, agent_id)
                                          for agent_id in self.ids))
            lp_chosens = dict(zip(self.ids, lp_chosens))
            entropies = dict(zip(self.ids, entropies))
            return lp_chosens, entropies
        else:
            logits = obs[agent_id].matmul(self.actor_tabs[agent_id])
            lp = torch.nn.functional.log_softmax(logits, dim=2)
            lp_chosen = lp.gather(2, acts[agent_id])
            entropy = -lp.exp().mul(lp).sum(dim=2).unsqueeze(2)
            return lp_chosen, entropy

    def _update_critic(self, G, V, frozen=False):
        if frozen:
            with torch.no_grad():
                critic_loss = torch.nn.functional.mse_loss(G, V)
        else:
            critic_loss = torch.nn.functional.mse_loss(G, V)
            if self.clip_grad.get("critic"):
                torch.nn.utils.clip_grad_norm_(self.critic_tab, self.clip_grad["critic"])

        critic_metrics = {
            "critic_loss": critic_loss.item()
            }
        return critic_loss, critic_metrics

    def _update_actor(self, A, ratio, entropy, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A,
                                                                         ratio[agent_id],
                                                                         entropy[agent_id],
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            A_norm = (A-A.mean())/(A.std() + 1e-8)
            if frozen:
                with torch.no_grad():
                    policy_loss = torch.min(
                        A_norm * ratio,
                        A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                        ).mean()
                    actor_loss = -(policy_loss + self.ent_coef*entropy.mean())
            else:
                policy_loss = torch.min(
                    A_norm * ratio,
                    A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                    ).mean()
                actor_loss = -(policy_loss + self.ent_coef*entropy.mean())
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_tabs[agent_id], self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy.mean().item(),
                }
            return actor_loss, actor_metrics

    def store_transition(self, transitions, agent_id=None):
        # I expect this to come in as a dict of individual agent Transitions
        obs = {agent_id: torch.FloatTensor(t.obs)
               for agent_id, t in transitions.items()}
        action = {agent_id: torch.LongTensor(t.action)
                  for agent_id, t in transitions.items()}
        # Assumes common-payoff
        reward = torch.FloatTensor(list(transitions.values())[0].reward)
        terminated = torch.BoolTensor(list(transitions.values())[0].terminated)
        truncated = torch.BoolTensor(list(transitions.values())[0].truncated)
        joint_obs = torch.FloatTensor(list(transitions.values())[0].obs)
        n_obs = torch.FloatTensor(list(transitions.values())[0].n_obs)
        transition = Transition(
            obs=obs,
            action=action,
            n_obs=n_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            joint_obs=joint_obs,
            )
        self.last_transition = transition
        self.last_n_transitions.push(transition)
        self.store_exp(joint_obs)

    def _stack_transitions(self):
        joint_obs = torch.stack(list(t.joint_obs for t in self.last_n_transitions))
        agent_obs = {agent_id: torch.stack(list(t.obs[agent_id] for t in self.last_n_transitions))
                     for agent_id in self.ids}
        actions = {agent_id: torch.stack(list(t.action[agent_id] for t in self.last_n_transitions))
                   for agent_id in self.ids}
        return joint_obs, agent_obs, actions

    def store_exp(self, obs):
        if self.exp_buffer_size == 0:
            pass
        elif len(self.exp_buffer) >= self.exp_buffer_size:
            if np.random.random() < self.exp_buffer_replacement_prob:
                idx = np.random.randint(self.exp_buffer_size)
                self.exp_buffer[idx] = obs
        else:
            self.exp_buffer.append(obs)

    def add_agent(
            self,
            agent_id,
            action_space: gym.Space,
            obs_space: gym.Space,
            actor_hidden_size,
            critic_hidden_size,
            actor_learning_rate,
            critic_learning_rate,
            gamma,
            ent_coef,
            n_steps,
            save_to_zoo=False
            ):
        ...

    @property
    def info(self):
        return dict(
            type="TabularMAPPO",
            train_steps=self.c_update*self.n_train_envs,
            actor_train_updates=self.total_actor_updates,
            critic_train_updates=self.total_critic_updates,
            actor_learning_rates=self.actor_learning_rates,
            critic_learning_rate=self.critic_learning_rate,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_epochs=self.n_epochs,
            clip_coef=self.clip_coef,
            ent_coef=self.ent_coef,
            clip_grad={"actor": self.clip_grad["actor"], "critic": self.clip_grad["critic"]},
            n_steps=self.n_steps,
            exp_buffer_size=self.exp_buffer_size,
            exp_buffer_replacement_prob=self.exp_buffer_replacement_prob,
            ignore_trunc=self.ignore_trunc
            )

    @classmethod
    def from_config(cls, cfg, env):
        agent_ids = [a["agent_id"] for a in cfg.agents]
        assert len(agent_ids) == cfg.n_agents, \
            f"len(agent_ids) = {len(agent_ids)} (expected {cfg.n_agents})"
        action_spaces = {
            agent_id: env.action_space
            for agent_id in agent_ids
            }
        obs_spaces = {
            agent_id: env.observation_space
            for agent_id in agent_ids
            }
        if hasattr(env, "joint_space"):
            joint_space = env.joint_space
        else:
            joint_space = env.observation_space
        actor_learning_rates = {
            agent_id: cfg.lr.actor
            for agent_id in agent_ids
            }
        critic_learning_rate = cfg.lr.critic
        gamma = cfg.get("gamma", 1.0)
        gae_lambda = cfg.get("gae_lambda", 1)
        n_epochs = cfg.get("n_epochs", 1)
        clip_coef = cfg.get("clip_coef", 0.1)
        ent_coef = cfg.get("ent_coef", 0.0)
        n_steps = cfg.get("n_steps", 5)
        clip_grad = cfg.get("clip_grad", {})
        save_to_zoo = cfg.get("save_to_zoo", False)
        ignore_trunc = cfg.get("ignore_trunc", True)
        if "exp_buffer" in cfg.keys():
            exp_buffer_size = cfg.exp_buffer.get("size", 0)
            exp_buffer_replacement_prob = cfg.exp_buffer.get("replacement_prob", 0.0)
        else:
            exp_buffer_size = 0
            exp_buffer_replacement_prob = 1.0

        if "freeze" in cfg.keys():
            freeze_critic = cfg.freeze.get("critic", 0)
            freeze_actor = cfg.freeze.get("actor", 0)
        else:
            freeze_critic = 0
            freeze_actor = 0

        if hasattr(env, "num_vec_envs"):
            n_train_envs = env.num_vec_envs
        elif hasattr(env, "vec_envs"):
            n_train_envs = len(env.vec_envs)
        else:
            n_train_envs = 1

        return cls(
            ids=agent_ids,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_epochs=n_epochs,
            clip_coef=clip_coef,
            ent_coef=ent_coef,
            n_steps=n_steps,
            clip_grad=clip_grad,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=save_to_zoo,
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=ignore_trunc,
            )

    def zoo_save(self, zoo_path, agent_save_names, env_cfg, evals, seed):
        agent_save_name_mapping = dict(zip(self.ids, agent_save_names))
        concat_name = "_".join(agent_save_names)
        # configs
        # actors
        # inelegant way of changing the evals into floats (from numpy.floats)
        # required so omegaconf can read
        float_evals = {
                "team": {k: v.item() for k, v in evals["team"].items()},
                "individual": {k: {k: v.item() for k, v in v.items()}
                               for k, v in evals["individual"].items()},
                }
        info = self.info
        for agent_id in self.ids:
            agent_name = agent_save_name_mapping[agent_id]
            model_pathname = osp.normpath(
                osp.join(zoo_path, "outputs", "actors", f"{agent_name}.pt")
                )
            config_pathname = osp.normpath(
                osp.join(zoo_path, "configs", "actors", f"{agent_name}.yaml")
                )
            os.makedirs(osp.dirname(model_pathname), exist_ok=True)
            os.makedirs(osp.dirname(config_pathname), exist_ok=True)
            torch.save(self.actor_tabs[agent_id], model_pathname)
            partner_agents = [agent_save_name_mapping[other_agent_id]
                              for other_agent_id in self.ids
                              if other_agent_id != agent_id]
            model_dict = dict(
                internal_name=agent_id,
                train_steps=info["train_steps"],
                train_updates=info["actor_train_updates"],
                learning_rate=info["actor_learning_rates"][agent_id],
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_epochs=self.n_epochs,
                clip_coef=self.clip_coef,
                ent_coef=self.ent_coef,
                clip_grad=self.clip_grad["actor"],
                n_steps=self.n_steps,
                ignore_trunc=self.ignore_trunc,
                )
            actor_dict = dict(
                name=agent_name,
                model=model_dict,
                train_env=env_cfg,
                eval={
                    "team": float_evals["team"],
                    "individual": float_evals["individual"][agent_id],
                    },
                critic=concat_name,
                experience=concat_name,
                partner_agents=partner_agents,
                path_to_model=model_pathname,
                seed=seed,
                )
            actor_dict = OmegaConf.create(actor_dict)
            with open(config_pathname, "w") as f:
                f.write(OmegaConf.to_yaml(actor_dict))

        # critic
        model_pathname = osp.normpath(
            osp.join(zoo_path, "outputs", "critics", f"{concat_name}.pt")
            )
        config_pathname = osp.normpath(
            osp.join(zoo_path, "configs", "critics", f"{concat_name}.yaml")
            )
        os.makedirs(osp.dirname(model_pathname), exist_ok=True)
        os.makedirs(osp.dirname(config_pathname), exist_ok=True)
        torch.save(self.critic_tab, model_pathname)
        model_dict = dict(
            train_steps=info["train_steps"],
            train_updates=info["critic_train_updates"],
            learning_rate=info["critic_learning_rate"],
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_epochs=self.n_epochs,
            clip_coef=self.clip_coef,
            clip_grad=self.clip_grad["critic"],
            n_steps=self.n_steps,
            ignore_trunc=self.ignore_trunc,
            )
        critic_dict = dict(
            name=concat_name,
            model=model_dict,
            train_env=env_cfg,
            eval=float_evals["team"],
            actors=agent_save_names,
            experience=concat_name,
            path_to_model=model_pathname,
            seed=seed,
            )
        critic_dict = OmegaConf.create(critic_dict)
        with open(config_pathname, "w") as f:
            f.write(OmegaConf.to_yaml(critic_dict))
        # experience
        experience_pathname = osp.normpath(
            osp.join(zoo_path, "outputs", "experience", f"{concat_name}.pt")
            )
        config_pathname = osp.normpath(
            osp.join(zoo_path, "configs", "experience", f"{concat_name}.yaml")
            )
        os.makedirs(osp.dirname(experience_pathname), exist_ok=True)
        os.makedirs(osp.dirname(config_pathname), exist_ok=True)
        if len(self.exp_buffer) > 0:
            save_buffer = torch.concat(self.exp_buffer)
        else:
            save_buffer = []
        torch.save(save_buffer, experience_pathname)
        buffer_dict = dict(
            max_size=self.exp_buffer_size,
            replacement_prob=self.exp_buffer_replacement_prob,
            size=len(self.exp_buffer),
            samples=len(save_buffer),
                )
        experience_dict = dict(
            name=concat_name,
            train_env=env_cfg,
            eval=float_evals["team"],
            experience=buffer_dict,
            critic=concat_name,
            actors=agent_save_names,
            path_to_experience=experience_pathname,
            seed=seed,
            )
        experience_dict = OmegaConf.create(experience_dict)
        with open(config_pathname, "w") as f:
            f.write(OmegaConf.to_yaml(experience_dict))
        return agent_save_name_mapping

    @classmethod
    def load_from_zoo(cls, cfg, name_id_mapping, env, zoo_path, reinit_critic=False):
        actors = {}
        agent_ids = list(name_id_mapping.values())
        critics = []
        exp_buffers = []
        for actor_name, agent_id in name_id_mapping.items():
            # load config
            path = osp.normpath(
                osp.join(zoo_path, "configs", "actors", f"{actor_name}.yaml")
                )
            actor_cfg = OmegaConf.load(path)
            actors[agent_id] = torch.load(actor_cfg.path_to_model)
            # load critic network
            critic_path = osp.normpath(
                osp.join(zoo_path, "configs", "critics", f"{actor_cfg.critic}.yaml")
                )
            critic_cfg = OmegaConf.load(critic_path)
            critics.append(torch.load(critic_cfg.path_to_model))
            # load experience
            exp_path = osp.normpath(
                osp.join(zoo_path, "configs", "experience", f"{actor_cfg.experience}.yaml")
                )
            exp_cfg = OmegaConf.load(exp_path)
            exp_buffers.append(torch.load(exp_cfg.path_to_experience))

        # Load env
        action_spaces = {
            agent_id: env.action_space
            for agent_id in agent_ids
            }
        obs_spaces = {
            agent_id: env.observation_space
            for agent_id in agent_ids
            }
        if hasattr(env, "joint_space"):
            joint_space = env.joint_space
        else:
            joint_space = env.observation_space

        critic = critics[0]  # Use the first critic

        actor_learning_rates = {
            agent_id: cfg.lr.actor
            for agent_id in agent_ids
            }
        critic_learning_rate = cfg.lr.critic
        if "exp_buffer" in cfg.keys():
            exp_buffer_size = cfg.exp_buffer.get("size", 0)
            exp_buffer_replacement_prob = cfg.exp_buffer.get("replacement_prob", 0.0)
        else:
            exp_buffer_size = 0
            exp_buffer_replacement_prob = 1.0
        if "freeze" in cfg.keys():
            freeze_critic = cfg.freeze.get("critic", 0)
            freeze_actor = cfg.freeze.get("actor", 0)
        else:
            freeze_critic = 0
            freeze_actor = 0

        if hasattr(env, "num_vec_envs"):
            n_train_envs = env.num_vec_envs
        elif hasattr(env, "vec_envs"):
            n_train_envs = len(env.vec_envs)
        else:
            n_train_envs = 1

        agents = TabularMAPPO.from_tables(
            actors=actors,
            critic=critic,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rate=critic_learning_rate,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            n_epochs=cfg.n_epochs,
            clip_coef=cfg.clip_coef,
            ent_coef=cfg.ent_coef,
            n_steps=cfg.n_steps,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=cfg.get("save_to_zoo", False),
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=cfg.get("ignore_trunc", True)
            )
        return agents

    @classmethod
    def from_tables(
            cls,
            actors={},
            critic=None,
            action_spaces={},
            obs_spaces={},
            joint_space=None,
            actor_learning_rates={},
            critic_learning_rate=None,
            gamma=1.0,
            gae_lambda=1.0,
            n_epochs=1,
            clip_coef=0.1,
            ent_coef=0.0,
            clip_grad={},
            n_steps=5,
            exp_buffer_size=0,
            exp_buffer_replacement_prob=1.0,
            save_to_zoo=False,
            freeze_critic=0,
            freeze_actor=0,
            n_train_envs=1,
            ignore_trunc=True,
            ):
        ids = [agent_id for agent_id in actors.keys()]
        agent = cls(
            ids=ids,
            action_spaces=action_spaces,
            obs_spaces=obs_spaces,
            joint_space=joint_space,
            actor_learning_rates=actor_learning_rates,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_epochs=n_epochs,
            clip_coef=clip_coef,
            ent_coef=ent_coef,
            clip_grad=clip_grad,
            n_steps=n_steps,
            exp_buffer_size=exp_buffer_size,
            exp_buffer_replacement_prob=exp_buffer_replacement_prob,
            save_to_zoo=save_to_zoo,
            freeze_critic=freeze_critic,
            freeze_actor=freeze_actor,
            n_train_envs=n_train_envs,
            ignore_trunc=ignore_trunc,
            )

        # update the networks
        params = []
        for agent_id in ids:
            agent.actor_tabs[agent_id] = copy(actors[agent_id])
            params.append({
                "params": agent.actor_tabs[agent_id],
                "lr": actor_learning_rates[agent_id],
                })
            agent.critic_tab = copy(critic)
            params.append({
                "params": agent.critic_tab,
                "lr": critic_learning_rate,
                })
        agent.optim = Adam(params,
                           lr=1e-4,
                           eps=1e-5)
        agent.saveables.update({
            "critic": agent.critic_tab,
            "actor": agent.actor_tabs,
            "optim": agent.optim,
            })
        return agent



class BehaviourPriorIPPO(IPPO):
    # Using a fixed behaviour prior (i.e.: no updating)
    def __init__(self, *args, **kwargs):
        if (len(args)>0) and (issubclass(type(args[0]), IPPO)):
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(*args, **kwargs)
        self.behaviour_priors = kwargs.get("behaviour_priors", {})
        self.kl_coef = kwargs.get("kl_coef", self.ent_coef)

    def _compute_policy_qty(self, obs, acts, agent_id=None):
        if agent_id is None:
            lp_chosens, entropies, kls = zip(*(self._compute_policy_qty(obs, acts, agent_id)
                                          for agent_id in self.ids))
            lp_chosens = dict(zip(self.ids, lp_chosens))
            entropies = dict(zip(self.ids, entropies))
            kls = dict(zip(self.ids, kls))
            return lp_chosens, entropies, kls
        else:
            logits = self.actor_nets[agent_id](obs[agent_id])
            bp_logits = self.behaviour_priors[agent_id](obs[agent_id])
            lp = torch.nn.functional.log_softmax(logits, dim=2)
            bp_lp = torch.nn.functional.log_softmax(bp_logits, dim=2)
            lp_chosen = lp.gather(2, acts[agent_id])
            entropy = -lp.exp().mul(lp).sum(dim=2, keepdim=True)
            kl = lp.exp().mul(lp-bp_lp).sum(dim=2, keepdim=True)
            return lp_chosen, entropy, kl

    def _update_actor(self, A, ratio, entropy, kl, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A[agent_id],
                                                                         ratio[agent_id],
                                                                         entropy[agent_id],
                                                                         kl[agent_id],
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            A_norm = (A-A.mean())/(A.std() + 1e-8)
            if frozen:
                with torch.no_grad():
                    policy_loss = torch.min(
                        A_norm * ratio,
                        A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                        ).mean()
                    actor_loss = -(policy_loss + self.ent_coef*entropy.mean() - self.kl_coef*kl.mean())
            else:
                policy_loss = torch.min(
                    A_norm * ratio,
                    A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                    ).mean()
                actor_loss = -(policy_loss + self.ent_coef*entropy.mean() - self.kl_coef*kl.mean())
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_nets[agent_id].parameters(), self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy.mean().item(),
                "kl": kl.mean().item(),
                }
            return actor_loss, actor_metrics

    def update(self):
        self.c_update += 1
        critic_frozen = self.c_update <= self.freeze_critic
        actor_frozen = self.c_update <= self.freeze_actor
        if (critic_frozen and actor_frozen) or self.c_update % self.p_update != 0:
            return {}
        # generate G, V, lp_chosen, entropy
        _, agent_obs, actions = self._stack_transitions()
        As, Gs, Vs = self._compute_gae(agent_obs)
        with torch.no_grad():
            orig_lp_chosens, _, _ = self._compute_policy_qty(agent_obs, actions)
        # update loop:
        for epoch in range(self.n_epochs):
            self.optim.zero_grad()
            new_lp_chosens, new_entropies, new_kls = self._compute_policy_qty(agent_obs, actions)
            policy_ratio = {agent_id: torch.exp(new_lp_chosens[agent_id] - orig_lp_chosens[agent_id])
                            for agent_id in self.ids}
            new_Vs = {agent_id: self.critic_nets[agent_id](agent_obs[agent_id])
                      for agent_id in self.ids}
            critic_loss, critic_metrics = self._update_critic(Gs, new_Vs, frozen=critic_frozen)
            actor_loss, actor_metrics = self._update_actor(As, policy_ratio, new_entropies, new_kls, frozen=actor_frozen)
            total_loss = critic_loss + actor_loss
            total_loss.backward()
            self.optim.step()
            if not critic_frozen:
                self.total_critic_updates += 1
                for agent_id in self.ids:
                    for i, p in enumerate(self.critic_nets[agent_id].parameters()):
                        critic_metrics[agent_id][f"critic_grad_{i}"] = p.grad.detach().norm().item()
            if not actor_frozen:
                self.total_actor_updates += 1
                for agent_id in self.ids:
                    for i, p in enumerate(self.actor_nets[agent_id].parameters()):
                        actor_metrics[agent_id][f"actor_grad_{i}"] = p.grad.detach().norm().item()

        # return
        train_metrics = {
            **critic_metrics,
            **actor_metrics
            }
        return train_metrics

    @classmethod
    def bp_from_zoo(cls, cfg, name_id_mapping, env, zoo_path, reset=False):
        ippo_agents = cls.load_from_zoo(cfg, name_id_mapping, env, zoo_path)
        agents = cls(ippo_agents)
        agents.behaviour_priors = {agent_id: deepcopy(actor)
                                   for agent_id, actor in agents.actor_nets.items()}
        agents.kl_coef = cfg.get("kl_coef", 1.0)
        if reset:
            for agent_id in agents.ids:
                for layer in agents.actor_nets[agent_id].children():
                    layer.reset_parameters()
                for layer in agents.critic_nets[agent_id].children():
                    layer.reset_parameters()
        return agents

    @classmethod
    def from_config(cls, cfg, env):
        zoo_path = hydra.utils.to_absolute_path(cfg.get("zoo_path", "zoo"))
        name_id_mapping = {agent["zoo_name"]: agent["agent_id"]
                           for agent in cfg.agents}
        agents = cls.bp_from_zoo(
                cfg,
                name_id_mapping,
                env,
                zoo_path,
                reset=cfg.get("reset", False)
            )
        return agents


class BehaviourPriorTabularIPPO(TabularIPPO):
    # Using a fixed behaviour prior (i.e.: no updating)
    def __init__(self, *args, **kwargs):
        if (len(args)>0) and (issubclass(type(args[0]), TabularIPPO)):
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(*args, **kwargs)
        self.behaviour_priors = kwargs.get("behaviour_priors", {})
        self.kl_coef = kwargs.get("kl_coef", self.ent_coef)

    def _compute_policy_qty(self, obs, acts, agent_id=None):
        if agent_id is None:
            lp_chosens, entropies, kls = zip(*(self._compute_policy_qty(obs, acts, agent_id)
                                          for agent_id in self.ids))
            lp_chosens = dict(zip(self.ids, lp_chosens))
            entropies = dict(zip(self.ids, entropies))
            kls = dict(zip(self.ids, kls))
            return lp_chosens, entropies, kls
        else:
            logits = obs[agent_id].matmul(self.actor_tabs[agent_id])
            bp_logits = obs[agent_id].matmul(self.behaviour_priors[agent_id])
            lp = torch.nn.functional.log_softmax(logits, dim=2)
            bp_lp = torch.nn.functional.log_softmax(bp_logits, dim=2)
            lp_chosen = lp.gather(2, acts[agent_id])
            entropy = -lp.exp().mul(lp).sum(dim=2, keepdim=True)
            kl = lp.exp().mul(lp-bp_lp).sum(dim=2, keepdim=True)
            return lp_chosen, entropy, kl

    def _update_actor(self, A, ratio, entropy, kl, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A[agent_id],
                                                                         ratio[agent_id],
                                                                         entropy[agent_id],
                                                                         kl[agent_id],
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            A_norm = (A-A.mean())/(A.std() + 1e-8)
            if frozen:
                with torch.no_grad():
                    policy_loss = torch.min(
                        A_norm * ratio,
                        A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                        ).mean()
                    actor_loss = -(policy_loss + self.ent_coef*entropy.mean() - self.kl_coef*kl.mean())
            else:
                policy_loss = torch.min(
                    A_norm * ratio,
                    A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                    ).mean()
                actor_loss = -(policy_loss + self.ent_coef*entropy.mean() - self.kl_coef*kl.mean())
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_tabs[agent_id], self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy.mean().item(),
                "kl": kl.mean().item(),
                }
            return actor_loss, actor_metrics

    def update(self):
        self.c_update += 1
        critic_frozen = self.c_update <= self.freeze_critic
        actor_frozen = self.c_update <= self.freeze_actor
        if (critic_frozen and actor_frozen) or self.c_update % self.p_update != 0:
            return {}
        # generate G, V, lp_chosen, entropy
        _, agent_obs, actions = self._stack_transitions()
        As, Gs, Vs = self._compute_gae(agent_obs)
        with torch.no_grad():
            orig_lp_chosens, _, _ = self._compute_policy_qty(agent_obs, actions)
        # update loop:
        for epoch in range(self.n_epochs):
            self.optim.zero_grad()
            new_lp_chosens, new_entropies, new_kls = self._compute_policy_qty(agent_obs, actions)
            policy_ratio = {agent_id: torch.exp(new_lp_chosens[agent_id] - orig_lp_chosens[agent_id])
                            for agent_id in self.ids}
            new_Vs = {agent_id: agent_obs[agent_id].matmul(self.critic_tabs[agent_id]).unsqueeze(-1)
                      for agent_id in self.ids}
            critic_loss, critic_metrics = self._update_critic(Gs, new_Vs, frozen=critic_frozen)
            actor_loss, actor_metrics = self._update_actor(As, policy_ratio, new_entropies, new_kls, frozen=actor_frozen)
            total_loss = critic_loss + actor_loss
            total_loss.backward()
            self.optim.step()
            if not critic_frozen:
                self.total_critic_updates += 1
                for agent_id in self.ids:
                    critic_metrics[agent_id]["critic_grad"] = self.critic_tabs[agent_id].grad.detach().norm().item()
            if not actor_frozen:
                self.total_actor_updates += 1
                for agent_id in self.ids:
                    actor_metrics[agent_id]["actor_grad"] = self.actor_tabs[agent_id].grad.detach().norm().item()
        # return
        train_metrics = {
            **critic_metrics,
            **actor_metrics
            }
        return train_metrics

    @classmethod
    def bp_from_zoo(cls, cfg, name_id_mapping, env, zoo_path, reset=False):
        ippo_agents = cls.load_from_zoo(cfg, name_id_mapping, env, zoo_path)
        agents = cls(ippo_agents)
        agents.behaviour_priors = {agent_id: deepcopy(actor.detach())
                                   for agent_id, actor in agents.actor_tabs.items()}
        agents.kl_coef = cfg.get("kl_coef", 1.0)
        if reset:
            for agent_id in agents.ids:
                agents.actor_tabs[agent_id] = torch.zeros_like(
                    agents.actor_tabs[agent_id], requires_grad=True
                    )
                agents.critic_tabs[agent_id] = torch.zeros_like(
                    agents.critic_tabs[agent_id], requires_grad=True
                    )
        return agents

    @classmethod
    def from_config(cls, cfg, env):
        zoo_path = hydra.utils.to_absolute_path(cfg.get("zoo_path", "zoo"))
        name_id_mapping = {agent["zoo_name"]: agent["agent_id"]
                           for agent in cfg.agents}
        agents = cls.bp_from_zoo(
                cfg,
                name_id_mapping,
                env,
                zoo_path,
                reset=cfg.get("reset", False)
            )
        return agents


class BehaviourPriorDeIPPO(DeIPPO):
    # Using a fixed behaviour prior (i.e.: no updating)
    def __init__(self, *args, **kwargs):
        if (len(args)>0) and (issubclass(type(args[0]), IPPO)):
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(*args, **kwargs)
        self.behaviour_priors = kwargs.get("behaviour_priors", {})
        self.kl_coef = kwargs.get("kl_coef", self.ent_coef)

    def _compute_policy_qty(self, obs, acts, agent_id=None):
        if agent_id is None:
            lp_chosens, entropies, kls = zip(*(self._compute_policy_qty(obs, acts, agent_id)
                                          for agent_id in self.ids))
            lp_chosens = dict(zip(self.ids, lp_chosens))
            entropies = dict(zip(self.ids, entropies))
            kls = dict(zip(self.ids, kls))
            return lp_chosens, entropies, kls
        else:
            logits = self.actor_nets[agent_id](obs[agent_id])
            bp_logits = self.behaviour_priors[agent_id](obs[agent_id])
            lp = torch.nn.functional.log_softmax(logits, dim=2)
            bp_lp = torch.nn.functional.log_softmax(bp_logits, dim=2)
            lp_chosen = lp.gather(2, acts[agent_id])
            entropy = -lp.exp().mul(lp).sum(dim=2, keepdim=True)
            kl = lp.exp().mul(lp-bp_lp).sum(dim=2, keepdim=True)
            return lp_chosen, entropy, kl

    def _update_critic(self, G, V, imp_weight, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            critic_metrics = {}
            for agent_id in self.ids:
                critic_loss, critic_metrics[agent_id] = self._update_critic(G[agent_id], V[agent_id],
                                                                            imp_weight[agent_id],
                                                                            agent_id=agent_id,
                                                                            frozen=frozen)
                total_loss += critic_loss
            return total_loss, critic_metrics
        else:
            if frozen:
                with torch.no_grad():
                    critic_loss = (G-V).pow(2).mul(imp_weight).mean()
            else:
                #critic_loss = torch.nn.functional.mse_loss(G, V)
                critic_loss = (G-V).pow(2).mul(imp_weight).mean()
                if self.clip_grad.get("critic"):
                    torch.nn.utils.clip_grad_norm_(self.critic_nets[agent_id], self.clip_grad["critic"])
            critic_metrics = {
                "critic_loss": critic_loss.item()
                }
            return critic_loss, critic_metrics

    def _update_actor(self, A, ratio, entropy, kl, imp_weight, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A[agent_id],
                                                                         ratio[agent_id],
                                                                         entropy[agent_id],
                                                                         kl[agent_id],
                                                                         imp_weight[agent_id],
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            A_norm = (A-A.mean())/(A.std() + 1e-8)
            if frozen:
                with torch.no_grad():
                    policy_loss = imp_weight.mul(torch.min(
                        A_norm * ratio,
                        A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                        )).mean()
                    actor_loss = -(policy_loss + self.ent_coef*entropy.mean() - self.kl_coef*kl.mean())
            else:
                policy_loss = imp_weight.mul(torch.min(
                    A_norm * ratio,
                    A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                    )).mean()
                actor_loss = -(policy_loss + self.ent_coef*entropy.mean() - self.kl_coef*kl.mean())
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_nets[agent_id].parameters(), self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy.mean().item(),
                "kl": kl.mean().item(),
                "imp_weight/max": imp_weight.max().item(),
                "imp_weight/min": imp_weight.min().item(),
                "imp_weight/mean": imp_weight.mean().item(),
                }
            return actor_loss, actor_metrics

    def update(self):
        self.c_update += 1
        critic_frozen = self.c_update <= self.freeze_critic
        actor_frozen = self.c_update <= self.freeze_actor
        if (critic_frozen and actor_frozen) or self.c_update % self.p_update != 0:
            return {}
        # generate G, V, lp_chosen, entropy
        _, agent_obs, actions = self._stack_transitions()
        As, Gs, Vs = self._compute_gae(agent_obs)
        with torch.no_grad():
            behav_lp_chosens, _, _ = self._compute_policy_qty(agent_obs, actions)
            proxi_lp_chosens = behav_lp_chosens
        # update loop:
        for epoch in range(self.n_epochs):
            self.optim.zero_grad()
            new_lp_chosens, new_entropies, new_kls = self._compute_policy_qty(agent_obs, actions)
            policy_ratio = {agent_id: torch.exp(new_lp_chosens[agent_id] - proxi_lp_chosens[agent_id])
                            for agent_id in self.ids}
            imp_weight = {agent_id: torch.exp(proxi_lp_chosens[agent_id] - behav_lp_chosens[agent_id])
                          for agent_id in self.ids}
            new_Vs = {agent_id: self.critic_nets[agent_id](agent_obs[agent_id])
                      for agent_id in self.ids}
            critic_loss, critic_metrics = self._update_critic(Gs, new_Vs, imp_weight, frozen=critic_frozen)
            actor_loss, actor_metrics = self._update_actor(As, policy_ratio, new_entropies, new_kls, imp_weight, frozen=actor_frozen)
            total_loss = critic_loss + actor_loss
            total_loss.backward()
            self.optim.step()
            if not critic_frozen:
                self.total_critic_updates += 1
                for agent_id in self.ids:
                    for i, p in enumerate(self.critic_nets[agent_id].parameters()):
                        critic_metrics[agent_id][f"critic_grad_{i}"] = p.grad.detach().norm().item()
            if not actor_frozen:
                self.total_actor_updates += 1
                for agent_id in self.ids:
                    for i, p in enumerate(self.actor_nets[agent_id].parameters()):
                        actor_metrics[agent_id][f"actor_grad_{i}"] = p.grad.detach().norm().item()

        # return
        train_metrics = {
            **critic_metrics,
            **actor_metrics
            }
        return train_metrics

    @classmethod
    def bp_from_zoo(cls, cfg, name_id_mapping, env, zoo_path, reset=False):
        ippo_agents = cls.load_from_zoo(cfg, name_id_mapping, env, zoo_path)
        agents = cls(ippo_agents)
        agents.behaviour_priors = {agent_id: deepcopy(actor)
                                   for agent_id, actor in agents.actor_nets.items()}
        agents.kl_coef = cfg.get("kl_coef", 1.0)
        if reset:
            for agent_id in agents.ids:
                for layer in agents.actor_nets[agent_id].children():
                    layer.reset_parameters()
                for layer in agents.critic_nets[agent_id].children():
                    layer.reset_parameters()
        return agents

    @classmethod
    def from_config(cls, cfg, env):
        zoo_path = hydra.utils.to_absolute_path(cfg.get("zoo_path", "zoo"))
        name_id_mapping = {agent["zoo_name"]: agent["agent_id"]
                           for agent in cfg.agents}
        agents = cls.bp_from_zoo(
                cfg,
                name_id_mapping,
                env,
                zoo_path,
                reset=cfg.get("reset", False)
            )
        return agents


class BehaviourPriorTabularDeIPPO(TabularDeIPPO):
    # Using a fixed behaviour prior (i.e.: no updating)
    def __init__(self, *args, **kwargs):
        if (len(args)>0) and (issubclass(type(args[0]), TabularIPPO)):
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(*args, **kwargs)
        self.behaviour_priors = kwargs.get("behaviour_priors", {})
        self.kl_coef = kwargs.get("kl_coef", self.ent_coef)

    def _compute_policy_qty(self, obs, acts, agent_id=None):
        if agent_id is None:
            lp_chosens, entropies, kls = zip(*(self._compute_policy_qty(obs, acts, agent_id)
                                          for agent_id in self.ids))
            lp_chosens = dict(zip(self.ids, lp_chosens))
            entropies = dict(zip(self.ids, entropies))
            kls = dict(zip(self.ids, kls))
            return lp_chosens, entropies, kls
        else:
            logits = obs[agent_id].matmul(self.actor_tabs[agent_id])
            bp_logits = obs[agent_id].matmul(self.behaviour_priors[agent_id])
            lp = torch.nn.functional.log_softmax(logits, dim=2)
            bp_lp = torch.nn.functional.log_softmax(bp_logits, dim=2)
            lp_chosen = lp.gather(2, acts[agent_id])
            entropy = -lp.exp().mul(lp).sum(dim=2, keepdim=True)
            kl = lp.exp().mul(lp-bp_lp).sum(dim=2, keepdim=True)
            return lp_chosen, entropy, kl

    def _update_critic(self, G, V, imp_weight, frozen=False, agent_id=None):
        if agent_id is None:
            total_loss = 0
            critic_metrics = {}
            for agent_id in self.ids:
                critic_loss, critic_metrics[agent_id] = self._update_critic(G[agent_id], V[agent_id],
                                                                            imp_weight[agent_id],
                                                                            agent_id=agent_id,
                                                                            frozen=frozen)
                total_loss += critic_loss
            return total_loss, critic_metrics
        else:
            if frozen:
                with torch.no_grad():
                    critic_loss = (G-V).pow(2).mul(imp_weight).mean()
            else:
                critic_loss = (G-V).pow(2).mul(imp_weight).mean()
                if self.clip_grad.get("critic"):
                    torch.nn.utils.clip_grad_norm_(self.critic_tabs[agent_id], self.clip_grad["critic"])
            critic_metrics = {
                "critic_loss": critic_loss.item()
                }
            return critic_loss, critic_metrics

    def _update_actor(self, A, ratio, entropy, kl, imp_weight, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A[agent_id],
                                                                         ratio[agent_id],
                                                                         entropy[agent_id],
                                                                         kl[agent_id],
                                                                         imp_weight[agent_id],
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            A_norm = (A-A.mean())/(A.std() + 1e-8)
            if frozen:
                with torch.no_grad():
                    policy_loss = imp_weight.mul(torch.min(
                        A_norm * ratio,
                        A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                        )).mean()
                    actor_loss = -(policy_loss + self.ent_coef*entropy.mean() - self.kl_coef*kl.mean())
            else:
                policy_loss = imp_weight.mul(torch.min(
                    A_norm * ratio,
                    A_norm * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                    )).mean()
                actor_loss = -(policy_loss + self.ent_coef*entropy.mean() - self.kl_coef*kl.mean())
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_tabs[agent_id], self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy.mean().item(),
                "kl": kl.mean().item(),
                "imp_weight/max": imp_weight.max().item(),
                "imp_weight/min": imp_weight.min().item(),
                "imp_weight/mean": imp_weight.mean().item(),
                }
            return actor_loss, actor_metrics

    def update(self):
        self.c_update += 1
        critic_frozen = self.c_update <= self.freeze_critic
        actor_frozen = self.c_update <= self.freeze_actor
        if (critic_frozen and actor_frozen) or self.c_update % self.p_update != 0:
            return {}
        # generate G, V, lp_chosen, entropy
        _, agent_obs, actions = self._stack_transitions()
        As, Gs, Vs = self._compute_gae(agent_obs)
        with torch.no_grad():
            behav_lp_chosens, _, _ = self._compute_policy_qty(agent_obs, actions)
            proxi_lp_chosens = behav_lp_chosens
        # update loop:
        for epoch in range(self.n_epochs):
            self.optim.zero_grad()
            new_lp_chosens, new_entropies, new_kls = self._compute_policy_qty(agent_obs, actions)
            policy_ratio = {agent_id: torch.exp(new_lp_chosens[agent_id] - proxi_lp_chosens[agent_id])
                            for agent_id in self.ids}
            imp_weight = {agent_id: torch.exp(proxi_lp_chosens[agent_id] - behav_lp_chosens[agent_id])
                          for agent_id in self.ids}
            new_Vs = {agent_id: agent_obs[agent_id].matmul(self.critic_tabs[agent_id]).unsqueeze(-1)
                      for agent_id in self.ids}
            critic_loss, critic_metrics = self._update_critic(Gs, new_Vs, imp_weight, frozen=critic_frozen)
            actor_loss, actor_metrics = self._update_actor(As, policy_ratio, new_entropies, new_kls, imp_weight, frozen=actor_frozen)
            total_loss = critic_loss + actor_loss
            total_loss.backward()
            self.optim.step()
            if not critic_frozen:
                self.total_critic_updates += 1
                for agent_id in self.ids:
                    critic_metrics[agent_id]["critic_grad"] = self.critic_tabs[agent_id].grad.detach().norm().item()
            if not actor_frozen:
                self.total_actor_updates += 1
                for agent_id in self.ids:
                    actor_metrics[agent_id]["actor_grad"] = self.actor_tabs[agent_id].grad.detach().norm().item()
        # return
        train_metrics = {
            **critic_metrics,
            **actor_metrics
            }
        return train_metrics

    @classmethod
    def bp_from_zoo(cls, cfg, name_id_mapping, env, zoo_path, reset=False):
        ippo_agents = cls.load_from_zoo(cfg, name_id_mapping, env, zoo_path)
        agents = cls(ippo_agents)
        agents.behaviour_priors = {agent_id: deepcopy(actor.detach())
                                   for agent_id, actor in agents.actor_tabs.items()}
        agents.kl_coef = cfg.get("kl_coef", 1.0)
        if reset:
            for agent_id in agents.ids:
                agents.actor_tabs[agent_id] = torch.zeros_like(
                    agents.actor_tabs[agent_id], requires_grad=True
                    )
                agents.critic_tabs[agent_id] = torch.zeros_like(
                    agents.critic_tabs[agent_id], requires_grad=True
                    )
        return agents

    @classmethod
    def from_config(cls, cfg, env):
        zoo_path = hydra.utils.to_absolute_path(cfg.get("zoo_path", "zoo"))
        name_id_mapping = {agent["zoo_name"]: agent["agent_id"]
                           for agent in cfg.agents}
        agents = cls.bp_from_zoo(
                cfg,
                name_id_mapping,
                env,
                zoo_path,
                reset=cfg.get("reset", False)
            )
        return agents
