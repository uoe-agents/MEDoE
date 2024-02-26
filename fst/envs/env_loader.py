from gym_cooking.environment import cooking_zoo
import fst.envs.chainball.chainball as chainball
from fst.envs.agent_dict_concat_vecenv import agent_dict_concat_vec_env_v0
import vmas
import supersuit as ss
import numpy as np


def load_env(cfg, seed=0, eval_env=False):
    loader = ENV_REG.get(cfg.env.package, load_default)
    return loader(cfg, seed=seed, eval_env=eval_env)

def load_chainball(cfg, seed=0, eval_env=False):
    n_parallel = cfg.env.get("n_parallel", 1)
    parallelize_evaluation = cfg.run.get("parallelize_evaluation", True)
    n_cpus = cfg.env.get("n_cpus", 0)
    agents_cfg = cfg.agents.agents
    agent_ids = [a["agent_id"] for a in agents_cfg]
    p_env = chainball.parallel_env(loader="config", cfg=cfg.env)
    mve = ss.pettingzoo_env_to_vec_env_v1(p_env)
    cve = ss.concat_vec_envs_v1(mve, n_parallel, num_cpus=n_cpus)
    ade = agent_dict_concat_vec_env_v0(
        cve,
        agent_ids,
        raw_agent_ids=p_env.possible_agents,
        num_vec_envs=n_parallel,
        num_cpus=n_cpus,
        new_step_api=True
        )
    if eval_env:
        eval_p_env = chainball.parallel_env(loader="config", cfg=cfg.eval.env)
        eval_mve = ss.pettingzoo_env_to_vec_env_v1(eval_p_env)
        if parallelize_evaluation:
            eval_cve = ss.concat_vec_envs_v1(eval_mve,
                                             cfg.run.n_eval_episodes,
                                             num_cpus=n_cpus)
            eval_ade = agent_dict_concat_vec_env_v0(
                eval_cve,
                agent_ids,
                raw_agent_ids=eval_p_env.possible_agents,
                num_vec_envs=cfg.run.n_eval_episodes,
                num_cpus=n_cpus,
                new_step_api=True
                )
        else:
            eval_cve = ss.concat_vec_envs_v1(eval_mve, 1)
            eval_ade = agent_dict_concat_vec_env_v0(eval_cve, agent_ids, new_step_api=True)
        return ade, eval_ade
    return ade

def load_cooking(cfg, seed=0, eval_env=False):
    n_parallel = cfg.env.get("n_parallel", 1)
    parallelize_evaluation = cfg.run.get("parallelize_evaluation", True)
    n_cpus = cfg.env.get("n_cpus", 0)
    agents_cfg = cfg.agents.agents
    agent_ids = [a["agent_id"] for a in agents_cfg]
    n_agents = len(agent_ids)

    config_dict = dict(
        level=cfg.env.level,
        num_agents=n_agents,
        record=cfg.env.get("record", False),
        max_steps=cfg.env.get("max_episode_length", 100),
        recipes=cfg.env.get("recipes", ["TomatoSalad"]*n_agents),
        obs_spaces=cfg.env.get("obs_representations", ["feature_vector_nc"]),
        action_scheme=cfg.env.get("action_scheme", "full_action_scheme"),
        ghost_agents=cfg.env.get("ghost_agents", 0),
        completion_reward_frac=cfg.env.get("completion_reward_frac", 0.2),
        time_penalty=cfg.env.get("time_penalty", 0),
        ego_first=cfg.env.get("ego_first", True),
        )
    p_env = cooking_zoo.parallel_env(**config_dict)
    #p_env.seed(seed=seed)
    mve = ss.pettingzoo_env_to_vec_env_v1(p_env)
    cve = ss.concat_vec_envs_v1(mve, n_parallel, num_cpus=n_cpus)
    ade = agent_dict_concat_vec_env_v0(
        cve,
        agent_ids,
        raw_agent_ids=p_env.possible_agents,
        num_vec_envs=n_parallel,
        num_cpus=n_cpus,
        new_step_api=True
        )

    if eval_env:
        eval_config_dict = dict(
            level=cfg.eval.env.level,
            num_agents=n_agents,
            record=cfg.eval.env.get("record", False),
            max_steps=cfg.eval.env.get("max_episode_length", 100),
            recipes=cfg.eval.env.get("recipes", ["TomatoSalad"]*n_agents),
            obs_spaces=cfg.eval.env.get("obs_representations", ["feature_vector_nc"]),
            action_scheme=cfg.env.get("action_scheme", "full_action_scheme"),
            ghost_agents=cfg.env.get("ghost_agents", 0),
            completion_reward_frac=cfg.eval.env.get("completion_reward_frac", 0.2),
            time_penalty=cfg.env.get("time_penalty", 0),
            ego_first=cfg.eval.env.get("ego_first", True),
        )
        eval_p_env = cooking_zoo.parallel_env(**eval_config_dict)
        #eval_p_env.seed(seed=seed)
        eval_mve = ss.pettingzoo_env_to_vec_env_v1(eval_p_env)
        if parallelize_evaluation:
            eval_cve = ss.concat_vec_envs_v1(eval_mve,
                                             cfg.run.n_eval_episodes,
                                             num_cpus=n_cpus)
            eval_ade = agent_dict_concat_vec_env_v0(
                eval_cve,
                agent_ids,
                raw_agent_ids=eval_p_env.possible_agents,
                num_vec_envs=cfg.run.n_eval_episodes,
                num_cpus=n_cpus,
                new_step_api=True
                )
        else:
            eval_cve = ss.concat_vec_envs_v1(eval_mve, 1)
            eval_ade = agent_dict_concat_vec_env_v0(eval_cve, agent_ids, new_step_api=True)
        return ade, eval_ade

    return ade

def load_vmas(cfg, seed=0, eval_env=False):
    parallelize_evaluation = cfg.run.get("parallelize_evaluation", True)
    agents_cfg = cfg.agents.agents
    agent_ids = [a["agent_id"] for a in agents_cfg]
    scenario_name  = cfg.env.scenario
    ade = vmas.make_env(
            scenario=scenario_name,
            num_envs=cfg.env.get("n_parallel", 1),
            device="cpu",
            continuous_actions=False,
            wrapper=vmas.Wrapper.ADE,
            max_steps=cfg.env.get("max_episode_length", None),
            dict_spaces=True,
            seed=seed,
            **cfg.env.get("scenario_cfg", {}),
            )
    if eval_env:
        eval_ade = vmas.make_env(
                scenario=scenario_name,
                num_envs=cfg.run.n_eval_episodes if parallelize_evaluation else 1,
                device="cpu",
                continuous_actions=False,
                wrapper=vmas.Wrapper.ADE,
                max_steps=cfg.eval.env.get("max_episode_length", None),
                dict_spaces=True,
                seed=seed+65536,
                **cfg.eval.env.get("scenario_cfg", {}),
                )
        return ade, eval_ade
    return ade

def load_default(cfg):
    msg = """Package {cfg.env.package} not found in the environment registry.
    Available environments are {list(ENV_REG.keys())}
    """
    raise Exception(msg)


ENV_REG = {
        "cooking": load_cooking,
        "chainball": load_chainball,
        "vmas": load_vmas,
        }
