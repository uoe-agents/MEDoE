import os
import os.path as osp
from collections import OrderedDict
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from fst.utils.network import fc_network
from omegaconf import OmegaConf
import hydra

def doe_classifier_config_loader(cfg, ids):
    type = cfg.type
    registry = {
        "MLP": MLPClassifier,
        }
    cls = registry[type]
    if not hasattr(cls, "from_config"):
        raise NotImplementedError(f"There is no from_config method defined for {type}")
    else:
        return cls.from_config(ids, cfg)


class DoEClassifier:

    def __init__(self):
        ...

    def is_doe(self, obs, agent_id=None):
        ...

    def update(self):
        ...


class MLPClassifier(DoEClassifier):

    def __init__(self,
                 ids,
                 train_dataloader, 
                 test_dataloader,
                 network_arch,
                 agent_id_to_label,
                 learning_rate=1e-2,
                 batch_size=256,
                 test_period=5,
                 obs_mask=None,
                 ):
        self.ids = ids
        self.mlps = {}
        self.learning_rates = {}
        if train_dataloader is not None:
            self.train_data_loader = train_dataloader
            self.test_data_loader = test_dataloader
            self.results = self.train_mlp(
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    network_arch=network_arch,
                    agent_id_to_label=agent_id_to_label,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    test_period=test_period,
                    obs_mask=obs_mask,
                    )

    def train_mlp(
            self,
            train_dataloader,
            test_dataloader,
            network_arch,
            agent_id_to_label,
            learning_rate=1e-2,
            batch_size=256,
            test_period=5,
            obs_mask=None,
            agent_id=None):
        # This could be more efficient, if it updated all the clasisifiers simultaneously
        if agent_id is None:
            self.mlps = {}
            # Returns the test results (for logging)
            return {agent_id: self.train_mlp(train_dataloader=train_dataloader,
                                             test_dataloader=test_dataloader,
                                             network_arch=network_arch,
                                             agent_id_to_label=agent_id_to_label,
                                             learning_rate=learning_rate,
                                             batch_size=batch_size,
                                             test_period=test_period,
                                             obs_mask=obs_mask,
                                             agent_id=agent_id
                                             )
                    for agent_id in self.ids}
        else:
            params = []
            self.mlps[agent_id] = fc_network(network_arch)
            self.learning_rates[agent_id] = learning_rate
            params.append({
                "params": self.mlps[agent_id].parameters(),
                "lr": self.learning_rates[agent_id],
                })
            loss_function = torch.nn.BCEWithLogitsLoss()
            self.optim = Adam(params,
                              lr=params[0]["lr"],
                              eps=1e-8)

            train_results = {}
            test_results = {}

            if obs_mask is None:
                mask = 1
            else:
                mask = torch.zeros(network_arch[0])
                for i in obs_mask:
                    mask[i] = 1

            for batch, (s, label) in enumerate(train_dataloader):
                predicted_label = self.mlps[agent_id](s*mask).flatten()
                egocentric_label = (label == agent_id_to_label[agent_id]).float()
                loss = loss_function(predicted_label, egocentric_label)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                train_results[batch] = loss.item()
                test_loss = 0.0
                if batch % test_period == 0:
                    with torch.no_grad():
                        for s_test, label_test in test_dataloader:
                            predicted_label_test = self.mlps[agent_id](s_test).flatten()
                            egocentric_label_test = (label_test == agent_id_to_label[agent_id]).float()
                            test_loss += loss_function(predicted_label_test, egocentric_label_test).item()
                        test_results[batch] = test_loss/len(test_dataloader)
            return {
                "train": train_results,
                "test": test_results,
            }

    def is_doe(self, obs, agent_id=None):
        if agent_id is None:
            return {agent_id: self.is_doe(obs[agent_id], agent_id=agent_id)
                    for agent_id in self.ids}
        else:
            return self.mlps[agent_id](torch.Tensor(obs)).sigmoid()

    def is_doe_logits(self, obs, agent_id=None):
        if agent_id is None:
            return {agent_id: self.is_doe(obs[agent_id], agent_id=agent_id)
                    for agent_id in self.ids}
        else:
            return self.mlps[agent_id](torch.Tensor(obs))

    def update(self):
        ...

    def save(self, pathname):
        torch.save(self.mlps, pathname)

    @classmethod
    def from_zoo(cls,
                 name_id_mapping,
                 agent_id_to_label,
                 zoo_path,
                 cfg,
                 ):
        agent_ids = list(name_id_mapping.values())

            

        exp_buffers = {}
        for actor_name, agent_id in name_id_mapping.items():
            # load config
            actor_cfg_path = osp.normpath(
                osp.join(zoo_path, "configs", "actors", f"{actor_name}.yaml")
                )
            actor_cfg = OmegaConf.load(actor_cfg_path)
            # load experience
            exp_path = osp.normpath(
                osp.join(zoo_path, "configs", "experience", f"{actor_cfg.experience}.yaml")
                )
            exp_cfg = OmegaConf.load(exp_path)
            exp_buffers.update({agent_id: torch.load(exp_cfg.path_to_experience)})

        # Classifier training params
        batch_size = cfg.get("batch_size", 256)
        test_fraction = cfg.get("test_fraction", 0.1)
        hidden_sizes = cfg.get("hidden_sizes", [128])
        learning_rate = cfg.get("lr", 1e-2)
        test_period = cfg.get("test_period", 5)
        obs_mask = cfg.get("obs_mask", None)

        # Load & process the data
        states = []
        labels = []
        with torch.no_grad():
            for agent_id in agent_ids:
                #state = torch.concat(exp_buffers[agent_id])
                state = exp_buffers[agent_id]
                label = torch.full((len(exp_buffers[agent_id]),), agent_id_to_label[agent_id])
                states.append(state)
                labels.append(label)
            states = torch.concat(states)
            labels = torch.concat(labels)
            dataset = SimpleListDataset(states, labels)
            train_size = int(test_fraction * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                        [train_size, test_size])
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        network_arch = [states[0].size().numel(), *hidden_sizes, 1]
        return cls(
            ids=agent_ids,
            train_dataloader=train_dataloader, 
            test_dataloader=test_dataloader,
            network_arch=network_arch,
            agent_id_to_label=agent_id_to_label,
            learning_rate=learning_rate,
            batch_size=batch_size,
            test_period=test_period,
            obs_mask=obs_mask,
            )

    @classmethod
    def from_config(cls, ids, cfg):
        if cfg.load_mode == "train":
            classifier = cls.from_config_train(ids, cfg)
            if cfg.get("save_classifier", False):
                classifier.save(cfg.save_pathname)
            return classifier
        elif cfg.load_mode == "load":
            return cls.from_config_load(ids, cfg)

    @classmethod
    def from_config_train(cls, ids, cfg):
        name_id_mapping = OrderedDict(((cfg.zoo_mapping[agent_id], agent_id) for agent_id in ids))
        zoo_path = hydra.utils.to_absolute_path(cfg.get("zoo_path", "zoo"))
        mlp_cfg = cfg.mlp
        agent_id_to_label = {}
        shared_sources = cfg.shared_sources
        for label, (source, source_agent_ids) in enumerate(shared_sources.items()):
            for agent_id in source_agent_ids:
                agent_id_to_label[agent_id] = label
        return cls.from_zoo(name_id_mapping, agent_id_to_label, zoo_path, mlp_cfg)

    @classmethod
    def from_config_load(cls, ids, cfg):
        return cls.load_mlp(ids, cfg.path_to_classifier)

    @classmethod
    def load_mlp(cls, ids, pathname):
        classifier = cls(
                 ids,
                 train_dataloader=None, 
                 test_dataloader=None,
                 network_arch=None,
                 agent_id_to_label=None,
                 )
        classifier.mlps = torch.load(hydra.utils.to_absolute_path(pathname))
        return classifier


class JointMLPClassifier(DoEClassifier):

    def __init__(self):
        ...

    def is_doe(self, agent_id=None):
        ...

    def update(self):
        ...


class SimpleListDataset(Dataset):
        
    def __init__(self, x, y):
        # x
        if isinstance(x, torch.Tensor):
            self.x_train = torch.clone(x).detach().float()
        else:
            self.x_train = torch.tensor(x, dtype=torch.float32)
        # y
        if isinstance(y, torch.Tensor):
            self.y_train = torch.clone(y).detach().float()
        else:
            self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
