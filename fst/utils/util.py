import numpy as np
import torch
import os.path as osp
import time
from itertools import starmap


def to_onehot(i, N):
    return [int(i == n) for n in range(N)]


def act_dict_to_env(act, ids=[0, 1, 2]):
    """ids is an *ordered* list"""
    action_dim = 5
    env_act = [to_onehot(act[agent_id], action_dim)
               for agent_id in ids]
    return env_act


def lod_to_dol(lod, np_copy=False):
    if np_copy:
        return {key: np.copy([d[key] for d in lod])
                for key in lod[0].keys()}
    else:
        return {key: [d[key] for d in lod] for key in lod[0].keys()}


def dol_to_lod(dol):
    return [dict(zip(dol.keys(), v)) for v in zip(*dol.values())]


def random_name(rand_int=None, lower=True, seed=None, n_agents=1):
    dirpath = osp.dirname(__file__)
    if seed is None:
        seed = int(time.time_ns())
    rg = np.random.default_rng(seed=seed)
    fnpath = osp.join(dirpath, "firstnames.txt")
    snpath = osp.join(dirpath, "surnames.txt")
    with open(fnpath) as fn:
        firstnames = rg.choice(fn.readlines(), n_agents)
    with open(snpath) as sn:
        surnames = rg.choice(sn.readlines(), n_agents, replace=False)

    def combine_name(firstname, surname):
        if rand_int is None:
            fullname = f"{firstname.strip()}-{surname.strip()}"
        else:
            fullname = f"{firstname.strip()}-{surname.strip()}-{rg.integers(rand_int)}"
        return fullname.lower() if lower else fullname

    return list(starmap(combine_name, zip(firstnames, surnames)))


def softmax(x, T=1.0):
    if T == 0:
        # If temperature is zero, use 'hardmax'
        z = np.where(x == max(x), 1, 0)
        return z/sum(z)
    else:
        z = (x - max(x))/T
        return np.exp(z)/sum(np.exp(z))


def sigmoid(x):
    return np.where(x > 0,
                    1/(1 + np.exp(-x)),
                    np.exp(x)/(1 + np.exp(x))
                    )

def tt_to_done(terminated, truncated):
    if type(next(iter(terminated.values()))) is np.ndarray:
        return {agent_id: np.logical_or(terminated[agent_id], truncated[agent_id])
                for agent_id in terminated.keys()}
    else:
        return {agent_id: torch.logical_or(terminated[agent_id], truncated[agent_id])
                for agent_id in terminated.keys()}
