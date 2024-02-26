import os
import os.path as osp
from collections import defaultdict
import gym
import numpy as np
import torch
from torch.optim import Adam
from typing import Iterable
from fst.utils.experience import ReplayBuffer, Transition
from fst.agents.doe_classifier import doe_classifier_config_loader
from fst.agents.agents import Agent, MultiAgent
from fst.agents.actor_critic import IA2C, MAA2C, TabularIA2C, TabularMAA2C, \
                                    IPPO, MAPPO, TabularIPPO, TabularMAPPO, \
                                    DeIPPO, TabularDeIPPO, \
                                    BehaviourPriorIPPO, BehaviourPriorTabularIPPO, \
                                    BehaviourPriorDeIPPO, BehaviourPriorTabularDeIPPO
from fst.agents.doe_classifier import StateMapClassifier
from copy import copy, deepcopy
import hydra
from omegaconf import OmegaConf


class DoETabularIA2C(TabularIA2C):
    def __init__(self, *args, **kwargs):
        if type(args[0]) is TabularIA2C:
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(*args, **kwargs)

        self.ent_coef = 1.0 # needed to override the ent_coef called elsewhere

        self.base_temp = kwargs.get("base_temp", 1.0)
        self.base_lr = kwargs.get("base_lr", 1.0)
        self.base_ent = kwargs.get("base_ent", 1.0)

        self.boost_temp_coef = kwargs.get("boost_temp", 1.0)
        self.boost_lr_coef = kwargs.get("boost_lr", 1.0)
        self.boost_ent_coef = kwargs.get("boost_ent", 1.0)

        self.doe_classifier = doe_classifier_config_loader(
                cfg=kwargs.get("doe_classifier_cfg"),
                ids=self.ids
                )

    def is_doe(self, obs, agent_id=None):
        return self.doe_classifier.is_doe(obs, agent_id=agent_id)

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
            # Boost quantities according to DoE
            b_lp_chosen = lp_chosen * self.boost_lr(obs, agent_id=agent_id)
            b_entropy = entropy * self.boost_ent(obs, agent_id=agent_id)
            return b_lp_chosen, b_entropy

    def policy(self, obs, explore=True, agent_id=None, temp=1.0):
        with torch.no_grad():
            if agent_id is None:
                p = {agent_id: self.policy(obs[agent_id], explore=explore, agent_id=agent_id)
                     for agent_id in self.ids}
            else:
                obs_torch = torch.FloatTensor(obs[agent_id])
                logits = obs_torch.matmul(self.actor_tabs[agent_id])
                if explore:
                    p = torch.distributions.categorical.Categorical(logits=logits/temp)
                else:
                    p = torch.distributions.categorical.Categorical(logits=logits)
        return p

    def act(self, obs, explore=True, agent_id=None):
        if agent_id is None:
            return {agent_id: self.act(obs, explore=explore, agent_id=agent_id)
                    for agent_id in self.ids}
        else:
            policy = self.policy(obs,
                                 explore=explore,
                                 agent_id=agent_id,
                                 temp=self.boost_temp(obs, agent_id))
            return policy.sample().unsqueeze(1).numpy()

    def boost_lr(self, obs, agent_id=None):
        if agent_id is None:
            return {self.boost_lr(obs, agent_id) for agent_id in self.ids}
        else:
            boost = self.boost_lr_coef
            doe = self.is_doe(obs[agent_id], agent_id)
            return self.base_lr*(boost + (1-boost)*self.is_doe(obs[agent_id], agent_id))

    def boost_temp(self, obs, agent_id=None):
        if agent_id is None:
            return {self.boost_temp(obs, agent_id) for agent_id in self.ids}
        else:
            doe = self.is_doe(obs[agent_id], agent_id)
            return self.base_temp*torch.pow(self.boost_temp_coef, 1-doe)

    def boost_ent(self, obs, agent_id=None):
        if agent_id is None:
            return {self.boost_ent(obs, agent_id) for agent_id in self.ids}
        else:
            boost = self.boost_ent_coef
            return self.base_ent*(boost + (1-boost)*self.is_doe(obs[agent_id], agent_id))

    def reset_doe(self, reset_actor=True, reset_critic=True):
        params = []
        for agent_id in self.ids:
            if type(self.doe_classifier) is not StateMapClassifier:
                raise Exception(f"Reset not supported for {type(self.doe_classifier)} classifier")
            doe = self.doe_classifier.state_mapping
            if reset_actor:
                self.actor_tabs[agent_id] = doe.unsqueeze(-1)*self.actor_tabs[agent_id].detach()
                self.actor_tabs[agent_id].requires_grad_(True)
            if reset_critic:
                self.critic_tabs[agent_id] = doe*self.critic_tabs[agent_id].detach()
                self.critic_tabs[agent_id].requires_grad_(True)
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

    @classmethod
    def from_config(cls, cfg, env):
        zoo_path = hydra.utils.to_absolute_path(cfg.get("zoo_path", "zoo"))
        name_id_mapping = {agent["zoo_name"]: agent["agent_id"]
                           for agent in cfg.agents}
        use_medoe = ("medoe" in cfg.keys()) and cfg.medoe.get("enable_medoe", True)
        agents = TabularIA2C.load_from_zoo(
            cfg,
            name_id_mapping,
            env,
            zoo_path,
            )
        if not use_medoe:
            return agents
        agents = cls(
            agents,
            base_temp=cfg.medoe.base_vals.temp,
            base_ent=cfg.medoe.base_vals.ent,
            base_clip=cfg.medoe.base_vals.clip,
            #
            boost_temp=cfg.medoe.boost_vals.temp,
            boost_ent=cfg.medoe.boost_vals.ent,
            boost_clip=cfg.medoe.boost_vals.clip,
            #
            doe_classifier_cfg=cfg.medoe.classifier,
            )
        if cfg.medoe.get("reset_doe", False):
            if cfg.medoe.get("reset_only_actor", False):
                agents.reset_doe(reset_actor=True, reset_critic=False)
            else:
                agents.reset_doe(reset_actor=True, reset_critic=True)
        return agents


class DoEIA2C(IA2C):
    def __init__(self, *args, **kwargs):
        if type(args[0]) is IA2C:
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(*args, **kwargs)
        self.ent_coef = 1.0 # needed to override the ent_coef called elsewhere

        self.base_temp = kwargs.get("base_temp", 1.0)
        self.base_lr = kwargs.get("base_lr", 1.0)
        self.base_ent = kwargs.get("base_ent", 1.0)

        self.boost_temp_coef = kwargs.get("boost_temp", 1.0)
        self.boost_lr_coef = kwargs.get("boost_lr", 1.0)
        self.boost_ent_coef = kwargs.get("boost_ent", 1.0)

        self.doe_classifier = doe_classifier_config_loader(
                cfg=kwargs.get("doe_classifier_cfg"),
                ids=self.ids
                )

    def is_doe(self, obs, agent_id=None):
        return self.doe_classifier.is_doe(obs, agent_id=agent_id)

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
            # Boost quantities according to DoE
            b_lp_chosen = lp_chosen * self.boost_lr(obs, agent_id=agent_id)
            b_entropy = entropy * self.boost_ent(obs, agent_id=agent_id)
            return b_lp_chosen, b_entropy

    def policy(self, obs, explore=True, agent_id=None, temp=1.0):
        with torch.no_grad():
            if agent_id is None:
                p = {agent_id: self.policy(obs[agent_id], explore=explore, agent_id=agent_id)
                     for agent_id in self.ids}
            else:
                obs_torch = torch.FloatTensor(obs[agent_id])
                logits = self.actor_nets[agent_id](obs_torch)
                if explore:
                    p = torch.distributions.categorical.Categorical(logits=logits/temp)
                else:
                    p = torch.distributions.categorical.Categorical(logits=logits)
        return p

    def act(self, obs, explore=True, agent_id=None):
        if agent_id is None:
            return {agent_id: self.act(obs, explore=explore, agent_id=agent_id)
                    for agent_id in self.ids}
        else:
            policy = self.policy(obs,
                                 explore=explore,
                                 agent_id=agent_id,
                                 temp=self.boost_temp(obs, agent_id))
            return policy.sample().unsqueeze(1).numpy()

    def boost_lr(self, obs, agent_id=None):
        if agent_id is None:
            return {self.boost_lr(obs, agent_id) for agent_id in self.ids}
        else:
            boost = self.boost_lr_coef
            doe = self.is_doe(obs[agent_id], agent_id)
            return self.base_lr*(boost + (1-boost)*self.is_doe(obs[agent_id], agent_id))

    def boost_temp(self, obs, agent_id=None):
        if agent_id is None:
            return {self.boost_temp(obs, agent_id) for agent_id in self.ids}
        else:
            doe = self.is_doe(obs[agent_id], agent_id)
            return self.base_temp*torch.pow(self.boost_temp_coef, 1-doe)

    def boost_ent(self, obs, agent_id=None):
        if agent_id is None:
            return {self.boost_ent(obs, agent_id) for agent_id in self.ids}
        else:
            boost = self.boost_ent_coef
            return self.base_ent*(boost + (1-boost)*self.is_doe(obs[agent_id], agent_id))

    @classmethod
    def from_config(cls, cfg, env):
        zoo_path = hydra.utils.to_absolute_path(cfg.get("zoo_path", "zoo"))
        name_id_mapping = {agent["zoo_name"]: agent["agent_id"]
                           for agent in cfg.agents}
        use_medoe = ("medoe" in cfg.keys()) and cfg.medoe.get("enable_medoe", True)
        agents = IA2C.load_from_zoo(
            cfg,
            name_id_mapping,
            env,
            zoo_path,
            )
        if not use_medoe:
            return agents
        agents = cls(
            agents,
            base_temp=cfg.medoe.base_vals.temp,
            base_ent=cfg.medoe.base_vals.ent,
            base_clip=cfg.medoe.base_vals.clip,
            #
            boost_temp=cfg.medoe.boost_vals.temp,
            boost_ent=cfg.medoe.boost_vals.ent,
            boost_clip=cfg.medoe.boost_vals.clip,
            #
            doe_classifier_cfg=cfg.medoe.classifier,
            )
        return agents



class DoETabularIPPO(TabularIPPO):
    def __init__(self, *args, **kwargs):
        if type(args[0]) is TabularIPPO:
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(*args, **kwargs)

        self.base_temp = kwargs.get("base_temp", 1.0)
        self.base_clip = kwargs.get("base_clip", 1.0)
        self.base_ent = kwargs.get("base_ent", 1.0)

        self.boost_temp_coef = kwargs.get("boost_temp", 1.0)
        self.boost_clip_coef = kwargs.get("boost_clip", 1.0)
        self.boost_ent_coef = kwargs.get("boost_ent", 1.0)

        self.doe_classifier = doe_classifier_config_loader(
                cfg=kwargs.get("doe_classifier_cfg"),
                ids=self.ids
                )

    def is_doe(self, obs, agent_id=None):
        return self.doe_classifier.is_doe(obs, agent_id=agent_id)

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
            doe = {agent_id: self.is_doe(agent_obs[agent_id], agent_id)
                   for agent_id in self.ids}
        # update loop:
        for epoch in range(self.n_epochs):
            self.optim.zero_grad()
            new_lp_chosens, new_entropies = self._compute_policy_qty(agent_obs, actions)
            policy_ratio = {agent_id: torch.exp(new_lp_chosens[agent_id] - orig_lp_chosens[agent_id])
                            for agent_id in self.ids}
            new_Vs = {agent_id: agent_obs[agent_id].matmul(self.critic_tabs[agent_id]).unsqueeze(-1)
                      for agent_id in self.ids}
            critic_loss, critic_metrics = self._update_critic(Gs, new_Vs, frozen=critic_frozen)
            actor_loss, actor_metrics = self._update_actor(As, policy_ratio, new_entropies, doe=doe, frozen=actor_frozen)
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

    def _update_actor(self, A, ratio, entropy, doe=None, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A[agent_id],
                                                                         ratio[agent_id],
                                                                         entropy[agent_id],
                                                                         doe=doe[agent_id],
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            A_norm = (A-A.mean())/(A.std() + 1e-8)
            boosted_ent_coef = self.base_ent * torch.pow(self.boost_ent_coef, 1-doe)
            boosted_clip_coef = self.base_clip * torch.pow(self.boost_clip_coef, 1-doe)
            if frozen:
                with torch.no_grad():
                    clamped_ratio = torch.clamp(ratio, 1-boosted_clip_coef, 1+boosted_clip_coef)
                    policy_loss = torch.min(
                        A_norm * ratio,
                        A_norm * clamped_ratio,
                        ).mean()
                    actor_loss = -(policy_loss + entropy.mul(boosted_ent_coef).mean())
            else:
                clamped_ratio = torch.clamp(ratio, 1-boosted_clip_coef, 1+boosted_clip_coef)
                policy_loss = torch.min(
                    A_norm * ratio,
                    A_norm * clamped_ratio
                    ).mean()
                actor_loss = -(policy_loss + entropy.mul(boosted_ent_coef).mean())
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_tabs[agent_id], self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy.mean().item(),
                }
            return actor_loss, actor_metrics

    def policy(self, obs, explore=True, agent_id=None, temp=1.0):
        with torch.no_grad():
            if agent_id is None:
                p = {agent_id: self.policy(obs[agent_id], explore=explore, agent_id=agent_id)
                     for agent_id in self.ids}
            else:
                obs_torch = torch.FloatTensor(obs[agent_id])
                logits = obs_torch.matmul(self.actor_tabs[agent_id])
                if explore:
                    p = torch.distributions.categorical.Categorical(logits=logits/temp)
                else:
                    p = torch.distributions.categorical.Categorical(logits=logits)
        return p

    def act(self, obs, explore=True, agent_id=None):
        if agent_id is None:
            return {agent_id: self.act(obs, explore=explore, agent_id=agent_id)
                    for agent_id in self.ids}
        else:
            policy = self.policy(obs,
                                 explore=explore,
                                 agent_id=agent_id,
                                 temp=self.boost_temp(obs, agent_id))
            return policy.sample().unsqueeze(1).numpy()

    def boost_temp(self, obs, agent_id=None):
        if agent_id is None:
            return {self.boost_temp(obs, agent_id) for agent_id in self.ids}
        else:
            doe = self.is_doe(obs[agent_id], agent_id)
            return self.base_temp*torch.pow(self.boost_temp_coef, 1-doe)

    def reset_doe(self, reset_actor=True, reset_critic=True):
        params = []
        for agent_id in self.ids:
            if type(self.doe_classifier) is not StateMapClassifier:
                raise Exception(f"Reset not supported for {type(self.doe_classifier)} classifier")
            doe = self.doe_classifier.state_mapping
            if reset_actor:
                self.actor_tabs[agent_id] = doe.unsqueeze(-1)*self.actor_tabs[agent_id].detach()
                self.actor_tabs[agent_id].requires_grad_(True)
            if reset_critic:
                self.critic_tabs[agent_id] = doe*self.critic_tabs[agent_id].detach()
                self.critic_tabs[agent_id].requires_grad_(True)
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

    @classmethod
    def from_config(cls, cfg, env):
        zoo_path = hydra.utils.to_absolute_path(cfg.get("zoo_path", "zoo"))
        name_id_mapping = {agent["zoo_name"]: agent["agent_id"]
                           for agent in cfg.agents}
        use_medoe = ("medoe" in cfg.keys()) and cfg.medoe.get("enable_medoe", True)
        agents = TabularIPPO.load_from_zoo(
            cfg,
            name_id_mapping,
            env,
            zoo_path,
            )
        if not use_medoe:
            return agents
        agents = cls(
            agents,
            base_temp=cfg.medoe.base_vals.temp,
            base_ent=cfg.medoe.base_vals.ent,
            base_clip=cfg.medoe.base_vals.clip,
            #
            boost_temp=cfg.medoe.boost_vals.temp,
            boost_ent=cfg.medoe.boost_vals.ent,
            boost_clip=cfg.medoe.boost_vals.clip,
            #
            doe_classifier_cfg=cfg.medoe.classifier,
            )
        if cfg.medoe.get("reset_doe", False):
            if cfg.medoe.get("reset_only_actor", False):
                agents.reset_doe(reset_actor=True, reset_critic=False)
            else:
                agents.reset_doe(reset_actor=True, reset_critic=True)
        return agents


class DoEIPPO(IPPO):
    def __init__(self, *args, **kwargs):
        if type(args[0]) is IPPO:
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(*args, **kwargs)

        self.base_temp = kwargs.get("base_temp", 1.0)
        self.base_clip = kwargs.get("base_clip", 1.0)
        self.base_ent = kwargs.get("base_ent", 1.0)

        self.boost_temp_coef = kwargs.get("boost_temp", 1.0)
        self.boost_clip_coef = kwargs.get("boost_clip", 1.0)
        self.boost_ent_coef = kwargs.get("boost_ent", 1.0)

        self.doe_classifier = doe_classifier_config_loader(
                cfg=kwargs.get("doe_classifier_cfg"),
                ids=self.ids
                )

    def is_doe(self, obs, agent_id=None):
        return self.doe_classifier.is_doe(obs, agent_id=agent_id)

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
            doe = {agent_id: self.is_doe(agent_obs[agent_id], agent_id)
                   for agent_id in self.ids}
        # update loop:
        for epoch in range(self.n_epochs):
            self.optim.zero_grad()
            new_lp_chosens, new_entropies = self._compute_policy_qty(agent_obs, actions)
            policy_ratio = {agent_id: torch.exp(new_lp_chosens[agent_id] - orig_lp_chosens[agent_id])
                            for agent_id in self.ids}
            new_Vs = {agent_id: self.critic_nets[agent_id](agent_obs[agent_id])
                      for agent_id in self.ids}
            critic_loss, critic_metrics = self._update_critic(Gs, new_Vs, frozen=critic_frozen)
            actor_loss, actor_metrics = self._update_actor(As, policy_ratio, new_entropies, doe=doe, frozen=actor_frozen)
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

    def _update_actor(self, A, ratio, entropy, doe=None, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A[agent_id],
                                                                         ratio[agent_id],
                                                                         entropy[agent_id],
                                                                         doe=doe[agent_id],
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            A_norm = (A-A.mean())/(A.std() + 1e-8)
            boosted_ent_coef = self.base_ent * torch.pow(self.boost_ent_coef, 1-doe)
            boosted_clip_coef = self.base_clip * torch.pow(self.boost_clip_coef, 1-doe)
            if frozen:
                with torch.no_grad():
                    clamped_ratio = torch.clamp(ratio, 1-boosted_clip_coef, 1+boosted_clip_coef)
                    policy_loss = torch.min(
                        A_norm * ratio,
                        A_norm * clamped_ratio,
                        ).mean()
                    actor_loss = -(policy_loss + entropy.mul(boosted_ent_coef).mean())
            else:
                clamped_ratio = torch.clamp(ratio, 1-boosted_clip_coef, 1+boosted_clip_coef)
                policy_loss = torch.min(
                    A_norm * ratio,
                    A_norm * clamped_ratio
                    ).mean()
                actor_loss = -(policy_loss + entropy.mul(boosted_ent_coef).mean())
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_nets[agent_id].parameters(), self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy.mean().item(),
                }
            return actor_loss, actor_metrics

    def policy(self, obs, explore=True, agent_id=None, temp=1.0):
        with torch.no_grad():
            if agent_id is None:
                p = {agent_id: self.policy(obs[agent_id], explore=explore, agent_id=agent_id)
                     for agent_id in self.ids}
            else:
                obs_torch = torch.FloatTensor(obs[agent_id])
                logits = self.actor_nets[agent_id](torch.FloatTensor(obs[agent_id]))
                if explore:
                    p = torch.distributions.categorical.Categorical(logits=logits/temp)
                else:
                    p = torch.distributions.categorical.Categorical(logits=logits)
        return p

    def act(self, obs, explore=True, agent_id=None):
        if agent_id is None:
            return {agent_id: self.act(obs, explore=explore, agent_id=agent_id)
                    for agent_id in self.ids}
        else:
            policy = self.policy(obs,
                                 explore=explore,
                                 agent_id=agent_id,
                                 temp=self.boost_temp(obs, agent_id))
            return policy.sample().unsqueeze(1).numpy()

    def boost_temp(self, obs, agent_id=None):
        if agent_id is None:
            return {self.boost_temp(obs, agent_id) for agent_id in self.ids}
        else:
            doe = self.is_doe(obs[agent_id], agent_id)
            return self.base_temp*torch.pow(self.boost_temp_coef, 1-doe)

    @classmethod
    def from_config(cls, cfg, env):
        zoo_path = hydra.utils.to_absolute_path(cfg.get("zoo_path", "zoo"))
        name_id_mapping = {agent["zoo_name"]: agent["agent_id"]
                           for agent in cfg.agents}
        use_medoe = ("medoe" in cfg.keys()) and cfg.medoe.get("enable_medoe", True)
        agents = IPPO.load_from_zoo(
            cfg,
            name_id_mapping,
            env,
            zoo_path,
            )
        if not use_medoe:
            return agents
        agents = cls(
            agents,
            base_temp=cfg.medoe.base_vals.temp,
            base_ent=cfg.medoe.base_vals.ent,
            base_clip=cfg.medoe.base_vals.clip,
            #
            boost_temp=cfg.medoe.boost_vals.temp,
            boost_ent=cfg.medoe.boost_vals.ent,
            boost_clip=cfg.medoe.boost_vals.clip,
            #
            doe_classifier_cfg=cfg.medoe.classifier,
            )
        return agents


class DoEDeIPPO(DeIPPO):
    def __init__(self, *args, **kwargs):
        if issubclass(type(args[0]), IPPO):
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(*args, **kwargs)

        self.base_temp = kwargs.get("base_temp", 1.0)
        self.base_clip = kwargs.get("base_clip", 1.0)
        self.base_ent = kwargs.get("base_ent", 1.0)

        self.boost_temp_coef = kwargs.get("boost_temp", 1.0)
        self.boost_clip_coef = kwargs.get("boost_clip", 1.0)
        self.boost_ent_coef = kwargs.get("boost_ent", 1.0)

        self.doe_classifier = doe_classifier_config_loader(
                cfg=kwargs.get("doe_classifier_cfg"),
                ids=self.ids
                )

    def is_doe(self, obs, agent_id=None):
        return self.doe_classifier.is_doe(obs, agent_id=agent_id)

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
            doe = {agent_id: self.is_doe(agent_obs[agent_id], agent_id)
                   for agent_id in self.ids}
            boost = self.boost_temp_coef
            temps = {agent_id: self.base_temp*(boost + (1-boost)*doe[agent_id])
                     for agent_id in self.ids}
            behav_lp_chosens, _ = self._compute_policy_qty(agent_obs, actions, temps)
            proxi_lp_chosens, _ = self._compute_policy_qty(agent_obs, actions)
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
            critic_loss, critic_metrics = self._update_critic(
                Gs, new_Vs, imp_weight, frozen=critic_frozen
                )
            actor_loss, actor_metrics = self._update_actor(
                As, policy_ratio, new_entropies, imp_weight, doe=doe, frozen=actor_frozen
                )
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

    def _compute_policy_qty(self, obs, acts, temp=defaultdict(lambda: 1), agent_id=None):
        if agent_id is None:
            lp_chosens, entropies = zip(*(self._compute_policy_qty(obs, acts, temp, agent_id)
                                          for agent_id in self.ids))
            lp_chosens = dict(zip(self.ids, lp_chosens))
            entropies = dict(zip(self.ids, entropies))
            return lp_chosens, entropies
        else:
            logits = self.actor_nets[agent_id](obs[agent_id])
            behav_logits = logits/temp[agent_id]
            lp = torch.nn.functional.log_softmax(logits, dim=2)
            behav_lp = torch.nn.functional.log_softmax(behav_logits, dim=2)
            lp_chosen = behav_lp.gather(2, acts[agent_id])
            entropy = -lp.exp().mul(lp).sum(dim=2).unsqueeze(2)
            return lp_chosen, entropy

    #def _update_actor(self, A, ratio, entropy, doe=None, agent_id=None, frozen=False):
    def _update_actor(self, A, ratio, entropy, imp_weight, doe=None, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A[agent_id],
                                                                         ratio[agent_id],
                                                                         entropy[agent_id],
                                                                         imp_weight[agent_id],
                                                                         doe=doe[agent_id],
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            A_norm = (A-A.mean())/(A.std() + 1e-8)
            boosted_ent_coef = self.base_ent * torch.pow(self.boost_ent_coef, 1-doe)
            boosted_clip_coef = self.base_clip * torch.pow(self.boost_clip_coef, 1-doe)
            if frozen:
                with torch.no_grad():
                    clamped_ratio = torch.clamp(ratio, 1-boosted_clip_coef, 1+boosted_clip_coef)
                    policy_loss = imp_weight.mul(torch.min(
                        A_norm * ratio,
                        A_norm * clamped_ratio,
                        )).mean()
                    actor_loss = -(policy_loss + entropy.mul(boosted_ent_coef).mean())
            else:
                clamped_ratio = torch.clamp(ratio, 1-boosted_clip_coef, 1+boosted_clip_coef)
                policy_loss = imp_weight.mul(torch.min(
                    A_norm * ratio,
                    A_norm * clamped_ratio
                    )).mean()
                actor_loss = -(policy_loss + entropy.mul(boosted_ent_coef).mean())
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_nets[agent_id].parameters(), self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy.mean().item(),
                }
            return actor_loss, actor_metrics

    def policy(self, obs, explore=True, agent_id=None, temp=1.0):
        with torch.no_grad():
            if agent_id is None:
                p = {agent_id: self.policy(obs[agent_id], explore=explore, agent_id=agent_id)
                     for agent_id in self.ids}
            else:
                obs_torch = torch.FloatTensor(obs[agent_id])
                logits = self.actor_nets[agent_id](torch.FloatTensor(obs[agent_id]))
                if explore:
                    p = torch.distributions.categorical.Categorical(logits=logits/temp)
                else:
                    p = torch.distributions.categorical.Categorical(logits=logits)
        return p

    def act(self, obs, explore=True, agent_id=None):
        if agent_id is None:
            return {agent_id: self.act(obs, explore=explore, agent_id=agent_id)
                    for agent_id in self.ids}
        else:
            policy = self.policy(obs,
                                 explore=explore,
                                 agent_id=agent_id,
                                 temp=self.boost_temp(obs, agent_id))
            return policy.sample().unsqueeze(1).numpy()

    def boost_temp(self, obs, agent_id=None):
        if agent_id is None:
            return {self.boost_temp(obs, agent_id) for agent_id in self.ids}
        else:
            doe = self.is_doe(obs[agent_id], agent_id)
            return self.base_temp*torch.pow(self.boost_temp_coef, 1-doe)

    @classmethod
    def from_config(cls, cfg, env):
        zoo_path = hydra.utils.to_absolute_path(cfg.get("zoo_path", "zoo"))
        name_id_mapping = {agent["zoo_name"]: agent["agent_id"]
                           for agent in cfg.agents}
        use_medoe = ("medoe" in cfg.keys()) and cfg.medoe.get("enable_medoe", True)
        agents = IPPO.load_from_zoo(
            cfg,
            name_id_mapping,
            env,
            zoo_path,
            )
        if not use_medoe:
            return agents
        agents = cls(
            agents,
            base_temp=cfg.medoe.base_vals.temp,
            base_ent=cfg.medoe.base_vals.ent,
            base_clip=cfg.medoe.base_vals.clip,
            #
            boost_temp=cfg.medoe.boost_vals.temp,
            boost_ent=cfg.medoe.boost_vals.ent,
            boost_clip=cfg.medoe.boost_vals.clip,
            #
            doe_classifier_cfg=cfg.medoe.classifier,
            )
        return agents


class DoETabularDeIPPO(TabularDeIPPO):
    def __init__(self, *args, **kwargs):
        if issubclass(type(args[0]), TabularIPPO):
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(*args, **kwargs)

        self.base_temp = kwargs.get("base_temp", 1.0)
        self.base_clip = kwargs.get("base_clip", 1.0)
        self.base_ent = kwargs.get("base_ent", 1.0)

        self.boost_temp_coef = kwargs.get("boost_temp", 1.0)
        self.boost_clip_coef = kwargs.get("boost_clip", 1.0)
        self.boost_ent_coef = kwargs.get("boost_ent", 1.0)

        self.doe_classifier = doe_classifier_config_loader(
                cfg=kwargs.get("doe_classifier_cfg"),
                ids=self.ids
                )

    def is_doe(self, obs, agent_id=None):
        return self.doe_classifier.is_doe(obs, agent_id=agent_id)

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
            doe = {agent_id: self.is_doe(agent_obs[agent_id], agent_id)
                   for agent_id in self.ids}
            boost = self.boost_temp_coef
            temps = {agent_id: self.base_temp*(boost + (1-boost)*doe[agent_id])
                     for agent_id in self.ids}
            behav_lp_chosens, _ = self._compute_policy_qty(agent_obs, actions, temps)
            proxi_lp_chosens, _ = self._compute_policy_qty(agent_obs, actions)
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
            critic_loss, critic_metrics = self._update_critic(
                Gs, new_Vs, imp_weight, frozen=critic_frozen
                )
            actor_loss, actor_metrics = self._update_actor(
                As, policy_ratio, new_entropies, imp_weight, doe=doe, frozen=actor_frozen
                )
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

    def _compute_policy_qty(self, obs, acts, temp=defaultdict(lambda: 1), agent_id=None):
        if agent_id is None:
            lp_chosens, entropies = zip(*(self._compute_policy_qty(obs, acts, temp, agent_id)
                                          for agent_id in self.ids))
            lp_chosens = dict(zip(self.ids, lp_chosens))
            entropies = dict(zip(self.ids, entropies))
            return lp_chosens, entropies
        else:
            logits = obs[agent_id].matmul(self.actor_tabs[agent_id])
            behav_logits = logits/temp[agent_id]
            lp = torch.nn.functional.log_softmax(logits, dim=2)
            behav_lp = torch.nn.functional.log_softmax(behav_logits, dim=2)
            lp_chosen = behav_lp.gather(2, acts[agent_id])
            entropy = -lp.exp().mul(lp).sum(dim=2).unsqueeze(2)
            return lp_chosen, entropy

    #def _update_actor(self, A, ratio, entropy, doe=None, agent_id=None, frozen=False):
    def _update_actor(self, A, ratio, entropy, imp_weight, doe=None, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A[agent_id],
                                                                         ratio[agent_id],
                                                                         entropy[agent_id],
                                                                         imp_weight[agent_id],
                                                                         doe=doe[agent_id],
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            A_norm = (A-A.mean())/(A.std() + 1e-8)
            boosted_ent_coef = self.base_ent * torch.pow(self.boost_ent_coef, 1-doe)
            boosted_clip_coef = self.base_clip * torch.pow(self.boost_clip_coef, 1-doe)
            if frozen:
                with torch.no_grad():
                    clamped_ratio = torch.clamp(ratio, 1-boosted_clip_coef, 1+boosted_clip_coef)
                    policy_loss = imp_weight.mul(torch.min(
                        A_norm * ratio,
                        A_norm * clamped_ratio,
                        )).mean()
                    actor_loss = -(policy_loss + entropy.mul(boosted_ent_coef).mean())
            else:
                clamped_ratio = torch.clamp(ratio, 1-boosted_clip_coef, 1+boosted_clip_coef)
                policy_loss = imp_weight.mul(torch.min(
                    A_norm * ratio,
                    A_norm * clamped_ratio
                    )).mean()
                actor_loss = -(policy_loss + entropy.mul(boosted_ent_coef).mean())
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_tabs[agent_id], self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy.mean().item(),
                }
            return actor_loss, actor_metrics

    def policy(self, obs, explore=True, agent_id=None, temp=1.0):
        with torch.no_grad():
            if agent_id is None:
                p = {agent_id: self.policy(obs[agent_id], explore=explore, agent_id=agent_id)
                     for agent_id in self.ids}
            else:
                obs_torch = torch.FloatTensor(obs[agent_id])
                logits = obs_torch.matmul(self.actor_tabs[agent_id])
                if explore:
                    p = torch.distributions.categorical.Categorical(logits=logits/temp)
                else:
                    p = torch.distributions.categorical.Categorical(logits=logits)
        return p


    def act(self, obs, explore=True, agent_id=None):
        if agent_id is None:
            return {agent_id: self.act(obs, explore=explore, agent_id=agent_id)
                    for agent_id in self.ids}
        else:
            policy = self.policy(obs,
                                 explore=explore,
                                 agent_id=agent_id,
                                 temp=self.boost_temp(obs, agent_id))
            return policy.sample().unsqueeze(1).numpy()

    def boost_temp(self, obs, agent_id=None):
        if agent_id is None:
            return {self.boost_temp(obs, agent_id) for agent_id in self.ids}
        else:
            doe = self.is_doe(obs[agent_id], agent_id)
            return self.base_temp*torch.pow(self.boost_temp_coef, 1-doe)

    def reset_doe(self, reset_actor=True, reset_critic=True):
        params = []
        for agent_id in self.ids:
            if type(self.doe_classifier) is not StateMapClassifier:
                raise Exception(f"Reset not supported for {type(self.doe_classifier)} classifier")
            doe = self.doe_classifier.state_mapping
            if reset_actor:
                self.actor_tabs[agent_id] = doe.unsqueeze(-1)*self.actor_tabs[agent_id].detach()
                self.actor_tabs[agent_id].requires_grad_(True)
            if reset_critic:
                self.critic_tabs[agent_id] = doe*self.critic_tabs[agent_id].detach()
                self.critic_tabs[agent_id].requires_grad_(True)
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

    @classmethod
    def from_config(cls, cfg, env):
        zoo_path = hydra.utils.to_absolute_path(cfg.get("zoo_path", "zoo"))
        name_id_mapping = {agent["zoo_name"]: agent["agent_id"]
                           for agent in cfg.agents}
        use_medoe = ("medoe" in cfg.keys()) and cfg.medoe.get("enable_medoe", True)
        agents = TabularIPPO.load_from_zoo(
            cfg,
            name_id_mapping,
            env,
            zoo_path,
            )
        if not use_medoe:
            return agents
        agents = cls(
            agents,
            base_temp=cfg.medoe.base_vals.temp,
            base_ent=cfg.medoe.base_vals.ent,
            base_clip=cfg.medoe.base_vals.clip,
            #
            boost_temp=cfg.medoe.boost_vals.temp,
            boost_ent=cfg.medoe.boost_vals.ent,
            boost_clip=cfg.medoe.boost_vals.clip,
            #
            doe_classifier_cfg=cfg.medoe.classifier,
            )
        if cfg.medoe.get("reset_doe", False):
            if cfg.medoe.get("reset_only_actor", False):
                agents.reset_doe(reset_actor=True, reset_critic=False)
            else:
                agents.reset_doe(reset_actor=True, reset_critic=True)
        return agents



class DoEBPIPPO(BehaviourPriorDeIPPO):
    def __init__(self, *args, **kwargs):
        if issubclass(type(args[0]), BehaviourPriorDeIPPO):
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(*args, **kwargs)

        self.base_temp = kwargs.get("base_temp", 1.0)
        self.base_clip = kwargs.get("base_clip", 1.0)
        self.base_ent = kwargs.get("base_ent", 1.0)
        self.base_kl = kwargs.get("base_kl", 1.0)

        self.boost_temp_coef = kwargs.get("boost_temp", 1.0)
        self.boost_clip_coef = kwargs.get("boost_clip", 1.0)
        self.boost_ent_coef = kwargs.get("boost_ent", 1.0)
        self.boost_kl_coef = kwargs.get("boost_kl", 1.0)

        self.doe_classifier = doe_classifier_config_loader(
                cfg=kwargs.get("doe_classifier_cfg"),
                ids=self.ids
                )

    def is_doe(self, obs, agent_id=None):
        return self.doe_classifier.is_doe(obs, agent_id=agent_id)

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
            doe = {agent_id: self.is_doe(agent_obs[agent_id], agent_id)
                   for agent_id in self.ids}
            boost = self.boost_temp_coef
            temps = {agent_id: self.base_temp*(boost + (1-boost)*doe[agent_id])
                     for agent_id in self.ids}
            behav_lp_chosens, _, _ = self._compute_policy_qty(agent_obs, actions, temps)
            proxi_lp_chosens, _, _ = self._compute_policy_qty(agent_obs, actions)
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
            critic_loss, critic_metrics = self._update_critic(
                Gs, new_Vs, imp_weight, frozen=critic_frozen
                )
            actor_loss, actor_metrics = self._update_actor(
                As, policy_ratio, new_entropies, new_kls, imp_weight, doe=doe, frozen=actor_frozen
                )
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

    def _compute_policy_qty(self, obs, acts, temp=defaultdict(lambda: 1), agent_id=None):
        if agent_id is None:
            lp_chosens, entropies, kls = zip(*(self._compute_policy_qty(obs, acts, temp, agent_id)
                                          for agent_id in self.ids))
            lp_chosens = dict(zip(self.ids, lp_chosens))
            entropies = dict(zip(self.ids, entropies))
            kls = dict(zip(self.ids, kls))
            return lp_chosens, entropies, kls
        else:
            logits = self.actor_nets[agent_id](obs[agent_id])
            behav_logits = self.actor_nets[agent_id](obs[agent_id])/temp[agent_id]
            bp_logits = self.behaviour_priors[agent_id](obs[agent_id])
            lp = torch.nn.functional.log_softmax(logits, dim=2)
            behav_lp = torch.nn.functional.log_softmax(behav_logits, dim=2)
            bp_lp = torch.nn.functional.log_softmax(bp_logits, dim=2)
            lp_chosen = behav_lp.gather(2, acts[agent_id])
            entropy = -lp.exp().mul(lp).sum(dim=2, keepdim=True)
            kl = lp.exp().mul(lp-bp_lp).sum(dim=2, keepdim=True)
            return lp_chosen, entropy, kl

    def _update_actor(self, A, ratio, entropy, kl, imp_weight, doe=None, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A[agent_id],
                                                                         ratio[agent_id],
                                                                         entropy[agent_id],
                                                                         kl[agent_id],
                                                                         imp_weight[agent_id],
                                                                         doe=doe[agent_id],
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            A_norm = (A-A.mean())/(A.std() + 1e-8)
            boosted_ent_coef = self.base_ent * torch.pow(self.boost_ent_coef, 1-doe)
            boosted_clip_coef = self.base_clip * torch.pow(self.boost_clip_coef, 1-doe)
            boosted_kl_coef = self.base_kl * torch.pow(self.boost_kl_coef, doe)
            if frozen:
                with torch.no_grad():
                    clamped_ratio = torch.clamp(ratio, 1-boosted_clip_coef, 1+boosted_clip_coef)
                    policy_loss = imp_weight.mul(torch.min(
                        A_norm * ratio,
                        A_norm * clamped_ratio,
                        )).mean()
                    actor_loss = -(policy_loss 
                                  + entropy.mul(boosted_ent_coef).mean() 
                                  - kl.mul(boosted_kl_coef).mean()
                                  )
            else:
                clamped_ratio = torch.clamp(ratio, 1-boosted_clip_coef, 1+boosted_clip_coef)
                policy_loss = imp_weight.mul(torch.min(
                    A_norm * ratio,
                    A_norm * clamped_ratio
                    )).mean()
                actor_loss = -(policy_loss 
                              + entropy.mul(boosted_ent_coef).mean() 
                              - kl.mul(boosted_kl_coef).mean()
                              )
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

    def policy(self, obs, explore=True, agent_id=None, temp=1.0):
        with torch.no_grad():
            if agent_id is None:
                p = {agent_id: self.policy(obs[agent_id], explore=explore, agent_id=agent_id)
                     for agent_id in self.ids}
            else:
                obs_torch = torch.FloatTensor(obs[agent_id])
                logits = self.actor_nets[agent_id](torch.FloatTensor(obs[agent_id]))
                if explore:
                    p = torch.distributions.categorical.Categorical(logits=logits/temp)
                else:
                    p = torch.distributions.categorical.Categorical(logits=logits)
        return p

    def act(self, obs, explore=True, agent_id=None):
        if agent_id is None:
            return {agent_id: self.act(obs, explore=explore, agent_id=agent_id)
                    for agent_id in self.ids}
        else:
            policy = self.policy(obs,
                                 explore=explore,
                                 agent_id=agent_id,
                                 temp=self.boost_temp(obs, agent_id))
            return policy.sample().unsqueeze(1).numpy()

    def boost_temp(self, obs, agent_id=None):
        if agent_id is None:
            return {self.boost_temp(obs, agent_id) for agent_id in self.ids}
        else:
            doe = self.is_doe(obs[agent_id], agent_id)
            return self.base_temp*torch.pow(self.boost_temp_coef, 1-doe)

    @classmethod
    def from_config(cls, cfg, env):
        zoo_path = hydra.utils.to_absolute_path(cfg.get("zoo_path", "zoo"))
        name_id_mapping = {agent["zoo_name"]: agent["agent_id"]
                           for agent in cfg.agents}
        use_medoe = ("medoe" in cfg.keys()) and cfg.medoe.get("enable_medoe", True)
        agents = BehaviourPriorDeIPPO.bp_from_zoo(
            cfg,
            name_id_mapping,
            env,
            zoo_path,
            )
        if not use_medoe:
            return agents
        agents = cls(
            agents,
            base_temp=cfg.medoe.base_vals.temp,
            base_ent=cfg.medoe.base_vals.ent,
            base_clip=cfg.medoe.base_vals.clip,
            base_kl=cfg.medoe.base_vals.kl,
            #
            boost_temp=cfg.medoe.boost_vals.temp,
            boost_ent=cfg.medoe.boost_vals.ent,
            boost_clip=cfg.medoe.boost_vals.clip,
            boost_kl=cfg.medoe.boost_vals.kl,
            #
            doe_classifier_cfg=cfg.medoe.classifier,
            )
        return agents


class DoEBPTabularIPPO(BehaviourPriorTabularDeIPPO):
    def __init__(self, *args, **kwargs):
        if issubclass(type(args[0]), BehaviourPriorTabularDeIPPO):
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(*args, **kwargs)

        self.base_temp = kwargs.get("base_temp", 1.0)
        self.base_clip = kwargs.get("base_clip", 1.0)
        self.base_ent = kwargs.get("base_ent", 1.0)
        self.base_kl = kwargs.get("base_kl", 1.0)

        self.boost_temp_coef = kwargs.get("boost_temp", 1.0)
        self.boost_clip_coef = kwargs.get("boost_clip", 1.0)
        self.boost_ent_coef = kwargs.get("boost_ent", 1.0)
        self.boost_kl_coef = kwargs.get("boost_kl", 1.0)

        self.doe_classifier = doe_classifier_config_loader(
                cfg=kwargs.get("doe_classifier_cfg"),
                ids=self.ids
                )
        # Temporary code to copy policies into code
        if kwargs.get("doe_classifier_cfg").type == "StateMapUpdate":
            self.doe_classifier.store_policies(self.actor_tabs)

    def is_doe(self, obs, agent_id=None):
        return self.doe_classifier.is_doe(obs, agent_id=agent_id)

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
            doe = self.is_doe(agent_obs)
            boost = self.boost_temp_coef
            temps = {agent_id: self.base_temp*(boost + (1-boost)*doe[agent_id])
                     for agent_id in self.ids}
            behav_lp_chosens, _, _ = self._compute_policy_qty(agent_obs, actions, temps)
            proxi_lp_chosens, _, _= self._compute_policy_qty(agent_obs, actions)
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
            critic_loss, critic_metrics = self._update_critic(
                Gs, new_Vs, imp_weight, frozen=critic_frozen
                )
            actor_loss, actor_metrics = self._update_actor(
                As, policy_ratio, new_entropies, new_kls, imp_weight, doe=doe, frozen=actor_frozen
                )
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

    def _compute_policy_qty(self, obs, acts, temp=defaultdict(lambda: 1), agent_id=None):
        if agent_id is None:
            lp_chosens, entropies, kls = zip(*(self._compute_policy_qty(obs, acts, temp, agent_id)
                                          for agent_id in self.ids))
            lp_chosens = dict(zip(self.ids, lp_chosens))
            entropies = dict(zip(self.ids, entropies))
            kls = dict(zip(self.ids, kls))
            return lp_chosens, entropies, kls
        else:
            logits = obs[agent_id].matmul(self.actor_tabs[agent_id])
            behav_logits = logits/temp[agent_id]
            bp_logits = obs[agent_id].matmul(self.behaviour_priors[agent_id])
            lp = torch.nn.functional.log_softmax(logits, dim=2)
            behav_lp = torch.nn.functional.log_softmax(behav_logits, dim=2)
            bp_lp = torch.nn.functional.log_softmax(bp_logits, dim=2)
            lp_chosen = behav_lp.gather(2, acts[agent_id])
            entropy = -lp.exp().mul(lp).sum(dim=2, keepdim=True)
            kl = lp.exp().mul(lp-bp_lp).sum(dim=2, keepdim=True)
            return lp_chosen, entropy, kl

    def _update_actor(self, A, ratio, entropy, kl, imp_weight, doe=None, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A[agent_id],
                                                                         ratio[agent_id],
                                                                         entropy[agent_id],
                                                                         kl[agent_id],
                                                                         imp_weight[agent_id],
                                                                         doe=doe[agent_id],
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            A_norm = (A-A.mean())/(A.std() + 1e-8)
            boosted_ent_coef = self.base_ent * torch.pow(self.boost_ent_coef, 1-doe)
            boosted_clip_coef = self.base_clip * torch.pow(self.boost_clip_coef, 1-doe)
            boosted_kl_coef = self.base_kl * torch.pow(self.boost_kl_coef, doe)
            if frozen:
                with torch.no_grad():
                    clamped_ratio = torch.clamp(ratio, 1-boosted_clip_coef, 1+boosted_clip_coef)
                    policy_loss = imp_weight.mul(torch.min(
                        A_norm * ratio,
                        A_norm * clamped_ratio,
                        )).mean()
                    actor_loss = -(policy_loss 
                                  + entropy.mul(boosted_ent_coef).mean() 
                                  - kl.mul(boosted_kl_coef).mean()
                                  )
            else:
                clamped_ratio = torch.clamp(ratio, 1-boosted_clip_coef, 1+boosted_clip_coef)
                policy_loss = imp_weight.mul(torch.min(
                    A_norm * ratio,
                    A_norm * clamped_ratio
                    )).mean()
                actor_loss = -(policy_loss 
                              + entropy.mul(boosted_ent_coef).mean() 
                              - kl.mul(boosted_kl_coef).mean()
                              )
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

    def policy(self, obs, explore=True, agent_id=None, temp=1.0):
        with torch.no_grad():
            if agent_id is None:
                p = {agent_id: self.policy(obs[agent_id], explore=explore, agent_id=agent_id)
                     for agent_id in self.ids}
            else:
                obs_torch = torch.FloatTensor(obs[agent_id])
                logits = obs_torch.matmul(self.actor_tabs[agent_id])
                if explore:
                    p = torch.distributions.categorical.Categorical(logits=logits/temp)
                else:
                    p = torch.distributions.categorical.Categorical(logits=logits)
        return p

    def act(self, obs, explore=True, agent_id=None):
        if agent_id is None:
            return {agent_id: self.act(obs, explore=explore, agent_id=agent_id)
                    for agent_id in self.ids}
        else:
            policy = self.policy(obs,
                                 explore=explore,
                                 agent_id=agent_id,
                                 temp=self.boost_temp(obs, agent_id))
            return policy.sample().unsqueeze(1).numpy()

    def boost_temp(self, obs, agent_id=None):
        if agent_id is None:
            return {self.boost_temp(obs, agent_id) for agent_id in self.ids}
        else:
            doe = self.is_doe(obs[agent_id], agent_id)
            return self.base_temp*torch.pow(self.boost_temp_coef, 1-doe)

    def reset_doe(self, reset_actor=True, reset_critic=True):
        params = []
        for agent_id in self.ids:
            if type(self.doe_classifier) is not StateMapClassifier:
                raise Exception(f"Reset not supported for {type(self.doe_classifier)} classifier")
            doe = self.doe_classifier.state_mapping
            if reset_actor:
                self.actor_tabs[agent_id] = doe.unsqueeze(-1)*self.actor_tabs[agent_id].detach()
                self.actor_tabs[agent_id].requires_grad_(True)
            if reset_critic:
                self.critic_tabs[agent_id] = doe*self.critic_tabs[agent_id].detach()
                self.critic_tabs[agent_id].requires_grad_(True)
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

    @classmethod
    def from_config(cls, cfg, env):
        zoo_path = hydra.utils.to_absolute_path(cfg.get("zoo_path", "zoo"))
        name_id_mapping = {agent["zoo_name"]: agent["agent_id"]
                           for agent in cfg.agents}
        use_medoe = ("medoe" in cfg.keys()) and cfg.medoe.get("enable_medoe", True)
        agents = BehaviourPriorTabularDeIPPO.bp_from_zoo(
            cfg,
            name_id_mapping,
            env,
            zoo_path,
            )
        if not use_medoe:
            return agents
        agents = cls(
            agents,
            base_temp=cfg.medoe.base_vals.temp,
            base_ent=cfg.medoe.base_vals.ent,
            base_clip=cfg.medoe.base_vals.clip,
            base_kl=cfg.medoe.base_vals.kl,
            #
            boost_temp=cfg.medoe.boost_vals.temp,
            boost_ent=cfg.medoe.boost_vals.ent,
            boost_clip=cfg.medoe.boost_vals.clip,
            boost_kl=cfg.medoe.boost_vals.kl,
            #
            doe_classifier_cfg=cfg.medoe.classifier,
            )
        if cfg.medoe.get("reset_doe", False):
            if cfg.medoe.get("reset_only_actor", False):
                agents.reset_doe(reset_actor=True, reset_critic=False)
            else:
                agents.reset_doe(reset_actor=True, reset_critic=True)
        return agents
