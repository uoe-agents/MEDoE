# Learning Complex Teamwork Tasks Using a Given Sub-task Decomposition

This repository contains code for the AAMAS 2024 paper ["Learning Complex Teamwork Tasks Using a Given Sub-task Decomposition"](http://arxiv.org/abs/2302.04944) by Fosong et. al.

## Requirements
To install dependencies, run
```bash
pip install -e .
```

QMIX experiments were performed using [EPyMARL](https://github.com/uoe-agents/epymarl). You may wish to use [this fork](https://github.com/efosong/epymarl/tree/vmas) to run VMAS efficiently via epymarl.

## Instructions
To run code, try
```bash
python scripts/simple_env_runner.py -cn CONFIG_NAME
```
The codebase uses [hydra](https://hydra.cc) for config management. Config files can be found in the `configs` directory.

Loading pre-trained agents from source tasks is done via an 'agent zoo' interface. An 'agent zoo' is a directory containing config files of agents, as well as saved models and experience buffers. Each agent in the zoo has a name, by which it can be referred. See [`configs/cooking_medoe.yaml`](https://github.com/uoe-agents/MEDoE/blob/main/fst/configs/cooking_medoe.yaml) for an example of a config file which loads zoo agents.  We provide the agent zoos used in this paper in our data upload (link TBA). 

## Results and Data Sharing
Experimental data, agent zoo directories, and plotting scripts can be found via the University of Edinburgh's DataShare service (Link TBA).

## Citation
```
@inproceedings{fosongLearningComplexTeamwork2024,
  title = {Learning Complex Teamwork Tasks Using a Given Sub-task Decomposition},
  booktitle = {Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems},
  author = {Fosong, Elliot and Rahman, Arrasy and Carlucho, Ignacio and Albrecht, Stefano V.},
  year = {2024},
  address = {Auckland, New Zealand},
}
```
