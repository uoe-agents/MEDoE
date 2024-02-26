import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from fst.runner.runner import EnvRunner
from fst.agents import agent_config_loader
from fst.agents.actor_critic import IPPO
from fst.envs.env_loader import load_env
from fst.utils.logger import Logger, AvgTracker
import wandb

from math import sqrt

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("sqrt", lambda x: sqrt(float(x)))
@hydra.main(version_base=None, config_path="../configs")
def run_exp(cfg: DictConfig):
    hcfg = hydra.core.hydra_config.HydraConfig.get()
    torch.set_num_threads(hcfg.launcher.get("cpus_per_task", 4))
    seed = cfg.get("seed", 0)
    zoo_path = hydra.utils.to_absolute_path(cfg.get("zoo_path", "zoo"))

    logger = Logger(cfg)
    _, env = load_env(cfg, seed=seed, eval_env=True)
    temp_agents = agent_config_loader(cfg.agents, env)
    runner = EnvRunner(
        env=env,
        eval_env=env,
        logger=logger,
        close_logger=False,
        close_env=False,
        agents=temp_agents,
        cfg=cfg,
        set_seed=True,
        zoo_path=""
        )
    checkpoint_dir = hydra.utils.to_absolute_path(cfg.checkpoint_dir)
    agent_id_translator = {
            a.agent_id: a.source_id
            for a in cfg.agents.agents
            }
    checkpoints = sorted(os.listdir(checkpoint_dir), key=checkpoint_to_num)
    checkpoint_returns = {}
    for checkpoint in checkpoints:
        checkpoint_return = run_one_checkpoint(
                cfg=cfg,
                env=env, 
                runner=runner,
                agent_id_translator=agent_id_translator,
                checkpoint_fname=os.path.join(checkpoint_dir, checkpoint),
                checkpoint_num=checkpoint_to_num(checkpoint),
                logger=logger
                )
        checkpoint_returns[checkpoint_to_num(checkpoint)] = checkpoint_return

    logger.close()
    torch.save(checkpoint_returns, "checkpoint_returns.pt")
    return checkpoint_returns

def checkpoint_to_num(checkpoint):
    return int(checkpoint.split("_")[1][1:])

def run_one_checkpoint(cfg, env, runner, agent_id_translator, checkpoint_fname, checkpoint_num, logger):
    checkpoint = torch.load(checkpoint_fname)
    agent_ids = agent_id_translator.keys()
    agents = IPPO.from_networks(
            actors={agent_id: checkpoint['actor'][agent_id_translator[agent_id]]
                    for agent_id in agent_ids},
            critics={agent_id: checkpoint['critic'][agent_id_translator[agent_id]]
                     for agent_id in agent_ids},
            action_spaces={agent_id: env.action_space for agent_id in agent_ids},
            obs_spaces={agent_id: env.observation_space for agent_id in agent_ids},
            joint_space=None,
            actor_learning_rates={agent_id: 1.0 for agent_id in agent_ids},
            critic_learning_rates={agent_id: 1.0 for agent_id in agent_ids},
            gamma=1.0,
            gae_lambda=1.0,
            n_epochs=1,
            clip_coef=0.1,
            ent_coef=0.0,
            n_steps=1,
            exp_buffer_size=0,
            exp_buffer_replacement_prob=0,
            save_to_zoo=False,
            freeze_critic=0,
            freeze_actor=0,
            n_train_envs=1,
            ignore_trunc=True,
        )
    team_returns = runner.evaluate(agents=agents)
    logger.wandb_log(
        "forget",
        np.mean(sum(team_returns.values())),
        step=checkpoint_num,
        commit=False
        )
    logger.logger.info(f"done checkpoint {checkpoint_num}: {np.mean(sum(team_returns.values()))}")
    return team_returns


if __name__ == "__main__":
    run_exp()
