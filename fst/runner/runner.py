from abc import ABC, abstractmethod
import time
from datetime import datetime
import os
import os.path as osp
from copy import deepcopy
import random
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from fst.utils.experience import Transition
from fst.utils.util import random_name, tt_to_done


class Runner(ABC):
    def __init__(
            self,
            config
            ):
        ...

    @abstractmethod
    def run(self):
        ...

class EnvRunner(Runner):
    """
    Multi-Agent Episodic Environment Runner
    For now we assume that we're using a multi-agent representation for the
    agents, e.g. via the MultiAgent class. We can still do single agent
    learning here by creating a MultiAgent object with just one agent.
    """
    def __init__(
            self,
            env,
            agents,
            cfg={},
            eval_env=None,
            logger=None,
            set_seed=False,
            model_save_dir="./saved_models",
            zoo_path="zoo",
            close_logger=True,
            close_env=True,
            ):
        self.env = env
        self.agents = agents
        self.cfg = cfg
        if logger is None:
            self.logger = Logger(cfg)
        else:
            self.logger = logger
        self.close_logger = close_logger
        self.close_env = close_env
        self.enable_render = cfg.logger.get("enable_render", True)
        self.max_training_steps = cfg.run.get("max_training_steps", np.Inf)
        self.max_training_episodes = cfg.run.get("max_training_episodes", np.Inf)
        if self.max_training_steps < 0:
            self.max_training_steps = np.Inf
        if self.max_training_episodes < 0:
            self.max_training_episodes = np.Inf

        self.n_eval_episodes = cfg.run.n_eval_episodes
        self.parallelize_evaluation = cfg.run.get("parallelize_evaluation", True)
        self.eval_env = eval_env
        self.zoo_path = zoo_path

        self.initialise_early_stop(cfg.run.early_stop)
        self.initialise_logger(cfg.logger, model_save_dir)

        if set_seed:
            torch.manual_seed(cfg.seed)
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)

    def initialise_early_stop(self, early_stop_cfg):
        self.last_eval_step = 0
        self.ema = 0
        self.c_above_threshold = 0
        self.early_stop_called = False
        self.do_early_stop = early_stop_cfg.get("do_early_stop", False)
        self.target_return = early_stop_cfg.get("target", 0.95)
        self.return_smoothing_coef = early_stop_cfg.get("smoothing", 0.31)
        self.target_return_hysteresis = early_stop_cfg.get("hysteresis", 0.05)
        self.steps_above_threshold = early_stop_cfg.get("steps", 50000)

    def initialise_logger(self, logger_cfg, model_save_dir):
        # Load the logging periods
        self.period = {
            "evaluate": logger_cfg.eval_period.val,
            "log_train": logger_cfg.train_metric_period.val,
            "save_model": logger_cfg.model_save_period.val,
            "save_render": logger_cfg.render_save_period.val,
            }
        self.period_units = {
            "evaluate": logger_cfg.eval_period.unit,
            "log_train": logger_cfg.train_metric_period.unit,
            "save_model": logger_cfg.model_save_period.unit,
            "save_render": logger_cfg.render_save_period.unit,
            }
        # Convert any 'frac' units to ep or step
        for qty in self.period.keys():
            if self.period_units[qty] == "step_frac":
                self.period[qty], self.period_units[qty] = \
                    round(self.period[qty] * self.max_training_steps), "step"
            if self.period_units[qty] == "ep_frac":
                self.period[qty], self.period_units[qty] = \
                    round(self.period[qty] * self.max_training_episodes), "ep"
        # Initialise counters
        self.counter = {
            "evaluate": 0,
            "log_train": 0,
            "save_model": 0,
            "save_render": 0
            }
        self.update = {
            "evaluate": False,
            "log_train": False,
            "save_model": False,
            "save_render": False
            }
        # Treat the time-based commit frequency as a special case
        self.wandb_commit_period = logger_cfg.get("wandb_commit_period", 60)
        # Misc settings
        if "eval_freq_boost" in logger_cfg.keys():
            self.eval_freq_boost_val = logger_cfg.eval_freq_boost.get("val", 10.0)
            self.eval_freq_boost_duration = logger_cfg.eval_freq_boost.get("duration", 0)
        else:
            self.eval_freq_boost_val = 10.0
            self.eval_freq_boost_duration = 0
        self.eval_initial_ep = logger_cfg.get("eval_initial_ep", True)
        self.eval_final_ep = logger_cfg.get("eval_final_ep", True)
        self.model_save_dir = model_save_dir
        if not os.path.exists(model_save_dir):
            os. makedirs(model_save_dir)

    def update_counter(self, counter_name, d_step, d_episode):
        freq_boost = self.eval_freq_boost_val if counter_name == "evaluate" else 1
        if self.period_units[counter_name] == "step":
            self.counter[counter_name] += d_step * freq_boost
        elif self.period_units[counter_name] == "ep":
            self.counter[counter_name] += d_episode * freq_boost
        self.update[counter_name] = \
            (self.counter[counter_name] >= self.period[counter_name])
        if self.update[counter_name]:
            self.counter[counter_name] = 0

    def update_counters(self, d_step, d_episode):
        for counter_name in self.counter.keys():
            self.update_counter(counter_name, d_step, d_episode)

    def update_eval_freq_boost(self, step):
        # For now this is a hardcoded threshold, but we could do something more sophisticated
        # e.g. eval_freq_boost = f(step)
        if step > self.eval_freq_boost_duration:
            self.eval_freq_boost_val = 1.0

    def evaluate(self, n_episodes=None, agents=None):
        if self.parallelize_evaluation:
            return self.evaluate_parallel(agents=agents)
        else:
            return self.evaluate_serial(n_episodes, agents=agents)

    def evaluate_serial(self, n_episodes, agents=None):
        if agents is None:
            agents = self.agents
        eval_returns = {agent_id: np.zeros(n_episodes)
                        for agent_id in agents.ids}
        step = 0
        episode = 0
        obs = self.eval_env.reset()
        while episode < n_episodes:
            # Action/Training loop
            actions = agents.act(obs, explore=False)
            n_obs, reward, terminated, truncated, info = self.eval_env.step(actions)
            done = tt_to_done(terminated, truncated)
            for agent_id in agents.ids:
                eval_returns[agent_id][episode] += reward[agent_id]
            step += 1
            episode += sum(next(iter(done.values())))
            obs = n_obs
        return eval_returns

    def evaluate_parallel(self, agents=None):
        if agents is None:
            agents = self.agents
        n_episodes = self.eval_env.num_vec_envs
        obs = self.eval_env.reset()
        if isinstance(next(iter(obs.values())), torch.Tensor):
            dones = torch.zeros(n_episodes, dtype=bool)
            eval_returns = {agent_id: torch.zeros(n_episodes)
                            for agent_id in agents.ids}
        else:
            dones = np.zeros(n_episodes, dtype=bool)
            eval_returns = {agent_id: np.zeros(n_episodes)
                            for agent_id in agents.ids}
        while not all(dones):
            actions = agents.act(obs, explore=False)
            n_obs, reward, terminated, truncated, info = self.eval_env.step(actions)
            for agent_id in agents.ids:
                eval_returns[agent_id] += ~dones * reward[agent_id]
            # here, we need to store <joint_action, reward, current_return> triplets
            dones = dones | next(iter(tt_to_done(terminated, truncated).values()))
            obs = n_obs
        return eval_returns

    def render(self, agents=None, render_mode="save", **kwargs):
        if agents is None:
            agents = self.agents
        ep_done = False
        obs = self.eval_env.reset()
        i = 0
        imgbuf = []
        mode = None
        while not ep_done:
            # Render
            if i == 0:
                # For some reason we need to render twice on the first step
                # if the rendering window already exists.
                self.eval_env.render()
            if render_mode == "save":
                if "ansi" in self.eval_env.metadata["render_modes"]:
                    mode = "ansi"
                    ansi_str = self.eval_env.render(mode="ansi")
                    imgbuf.append(ansi_str)
                elif "rgb_array" in self.eval_env.metadata["render_modes"]:
                    mode = "rgb_array"
                    img = self.eval_env.render(mode="rgb_array")
                    imgbuf.append(img)
            elif render_mode == "human":
                self.eval_env.render(mode=render_mode)
            # Action/Training loop
            actions = agents.act(obs,
                                 explore=kwargs.get("explore", False),
                                 )
            n_obs, reward, terminated, truncated, info = self.eval_env.step(actions)
            i+=1
            done = tt_to_done(terminated, truncated)
            ep_done = next(iter(done.values()))[0]
            obs = n_obs
        print(f"LENGTH: {len(imgbuf)}")
        if mode == "rgb_array":
            imgbuf = np.array(imgbuf, ndmin=5)
        return imgbuf, mode

    def run(self, explore=True):
        step = 0
        episode = 0
        obs = self.env.reset()
        train_metrics = {}
        team_returns = []


        # Log an initial set of episodes
        if self.eval_initial_ep:
            evals = self.log_eval(step, episode)
            self.save_model(step, episode)
            self.save_render(step, episode)
            team_returns.append(evals["team"]["mean"])

        while ((step < self.max_training_steps)
               and (episode < self.max_training_episodes)):
            # Action/Training loop
            actions = self.agents.act(obs, explore=explore)
            n_obs, reward, terminated, truncated, info = self.env.step(actions)
            done = tt_to_done(terminated, truncated)
            n_globs = self.env.global_obs()
            d_step = self.env.num_vec_envs
            step += d_step
            d_episode = sum(next(iter(done.values())))
            episode += d_episode
            # collect transition
            transitions = {agent_id: Transition(obs=obs[agent_id],
                                                action=actions[agent_id],
                                                n_obs=n_obs[agent_id],
                                                reward=reward[agent_id],
                                                terminated=terminated[agent_id],
                                                truncated=truncated[agent_id],
                                                joint_obs=None,
                                                )
                           for agent_id in self.agents.ids
                           }
            self.agents.store_transition(transitions)
            obs = n_obs
            train_metrics = self.agents.update() or train_metrics

            # Evaluation/Logging
            self.update_eval_freq_boost(step)
            self.update_counters(d_step, d_episode)
            if self.update["evaluate"]:
                evals = self.log_eval(step, episode)
                team_returns.append(evals["team"]["mean"])
                if self.early_stop(evals["team"]["mean"], step):
                    break
            if self.update["log_train"]:
                self.log_train(train_metrics, step)
            if self.update["save_model"]:
                self.save_model(step, episode)
            if self.update["save_render"]:
                self.save_render(step, episode)
            if time.time() - self.logger.time_last_commit > self.wandb_commit_period:
                self.logger.commit()

        # Log a final set of episodes before closing
        if self.eval_final_ep:
            evals = self.log_eval(step, episode)
            team_returns.append(evals["team"]["mean"])
            self.save_model(step, episode)
            self.save_render(step, episode)
            self.save_to_zoo(evals)
            # Overcooked:
            auc = np.mean(np.array(team_returns))
            if self.cfg.logger.get("log_overcooked_auc", False):
                self.logger.wandb_log("AUCGap",
                                      float(self.agents.n_agents-auc),
                                      step)
            # Chainball:
            if self.cfg.logger.get("log_chainball_auc", False):
                self.logger.wandb_log("AUCGap",
                                      1.0 - (auc+84.3)/(16.3+84.3),
                                      step)
            # VMAS:
            if self.cfg.logger.get("log_vmas_auc", False):
                self.logger.wandb_log("AUCGap",
                                      float(self.agents.n_agents-auc),
                                      step)

        if self.close_logger:
            self.logger.close()

        if self.close_env and hasattr(self.env, "close"):
            self.env.close()

        return team_returns

    def log_eval(self, step, episode, agents=None):
        eval_returns = self.evaluate(n_episodes=self.n_eval_episodes, agents=agents)
        evals = self.logger.log_eval_metrics(eval_returns, step)
        self.logger.log(episode, step, evals["team"]["mean"])
        return evals

    def log_train(self, train_metrics, step):
        self.logger.log_train_metrics(train_metrics, step)

    def save_model(self, step, episode):
        save_dirpath = os.path.join(
            self.model_save_dir,
            f"e{episode}_s{step}"
            )
        self.agents.save_params(save_dirpath)

    def save_render(self, step, episode, agents=None):
        if not self.enable_render:
            return

        video, mode = self.render(render_mode="save", agents=agents)
        if mode == "ansi":
            self.logger.log_ansi(video, step=step)

        elif mode == "rgb_array":
            if len(video) != 0:
                tag = f"Render/s{step}"
                self.logger.log_video(tag, video, step=step)

    def early_stop(self, mean_return, step):
        if not self.do_early_stop:
            return False
        # Work out how many steps since the last time early stop checked
        d_step = step - self.last_eval_step
        self.last_eval_step = step
        # Compute exponential moving average
        if not self.early_stop_called:
            self.early_stop_called = True
            self.ema = mean_return
        else:
            self.ema = ((1-self.return_smoothing_coef) * self.ema
                        + self.return_smoothing_coef * mean_return)
        # Test whether above (hysteresis) threshold
        previously_above_threshold = self.c_above_threshold > 0
        above_threshold = self.ema > self.target_return
        above_hysteresis_threshold = \
            self.ema > self.target_return - self.target_return_hysteresis
        if above_threshold:
            self.c_above_threshold += d_step
        elif previously_above_threshold and above_hysteresis_threshold:
            self.c_above_threshold += d_step
        else:
            self.c_above_threshold = 0
        return self.c_above_threshold > self.steps_above_threshold

    def save_to_zoo(self, evals, agents=None, cfg=None):
        if agents is None:
            agents = self.agents
        if cfg is None:
            cfg = self.cfg
        if not hasattr(agents, "zoo_save"):
            # Not implemented, fail silently
            return
        if not agents.save_to_zoo:
            # No request to save
            return
            
        agent_save_names = random_name(n_agents=len(agents.ids))
        agent_save_name_mapping = agents.zoo_save(
            self.zoo_path,
            agent_save_names,
            cfg.env,
            evals,
            cfg.seed
            )
        zoo_index_fname = osp.normpath(
            osp.join(self.zoo_path, f"index.csv")
            )
        os.makedirs(osp.dirname(zoo_index_fname), exist_ok=True)
        if not osp.exists(zoo_index_fname):
            csv_line = (
                "agent_id",
                "agent_type",
                "env_name",
                "tr_mean",
                "tr_std",
                "ir_mean",
                "ir_std",
                "zoo_path_root",
                "datetime",
                )
            with open(zoo_index_fname, "a") as f:
                f.write(",".join(csv_line)+'\n')

        for agent_id in agents.ids:
            csv_line = (
                agent_save_name_mapping[agent_id],
                agents.info["type"],
                cfg.env.name,
                str(evals["team"]["mean"]),
                str(evals["team"]["std"]),
                str(evals["individual"][agent_id]["mean"]),
                str(evals["individual"][agent_id]["std"]),
                self.zoo_path,
                datetime.now().isoformat(),
                )
            with open(zoo_index_fname, "a") as f:
                f.write(",".join(csv_line)+'\n')
