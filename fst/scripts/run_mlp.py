import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from fst.runner.runner import EnvRunner
from fst.agents import agent_config_loader
from fst.envs.env_loader import load_env
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
    team_returns = run_one_task(cfg, seed=seed, zoo_path=zoo_path)
    if len(team_returns) >= 3:
        output_metric = 0.67*team_returns[-1] + 0.24*team_returns[-2] + 0.09*team_returns[-3]
    else:
        output_metric = 0
    return float(output_metric)

def run_one_task(task_cfg, seed=0, zoo_path=""):
    env, eval_env = load_env(task_cfg, seed=seed, eval_env=True)
    env.global_obs = lambda: None
    agents = agent_config_loader(task_cfg.agents, env)
    runner = EnvRunner(
        env=env,
        eval_env=eval_env,
        agents=agents,
        cfg=task_cfg,
        set_seed=True,
        zoo_path=zoo_path,
        )
    torch.save(agents.doe_classifier.results, "train_results.pt")
    runner.logger.logger.info(f"TRAIN: {agents.doe_classifier.results}")
    team_returns = runner.run()
    return team_returns

if __name__ == "__main__":
    run_exp()
