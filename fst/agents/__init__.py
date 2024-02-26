from fst.agents.actor_critic import \
    A2C, \
    IA2C, MAA2C, \
    IPPO, MAPPO, \
    TabularIA2C, TabularMAA2C, \
    TabularIPPO, TabularMAPPO, \
    DeIPPO, TabularDeIPPO, \
    BehaviourPriorIPPO, BehaviourPriorTabularIPPO
from fst.agents.doe_ac import \
    DoEIA2C, \
    DoEIPPO,\
    DoETabularIA2C, \
    DoETabularIPPO, \
    DoEDeIPPO, DoETabularDeIPPO, \
    DoEBPIPPO, DoEBPTabularIPPO

def agent_config_loader(cfg, env):
    type = cfg.type
    registry = {
        # Actor Critic
        # -- Deep
        # ---- Vanilla
        "A2C": A2C,
        "IA2C": IA2C,
        "MAA2C": MAA2C,
        "IPPO": IPPO,
        "MAPPO": MAPPO,
        # ---- Decoupled
        "DeIPPO": DeIPPO,
        # ---- DoE
        "DoEIA2C": DoEIA2C,
        "DoEIPPO": DoEIPPO,
        "DoEDeIPPO": DoEDeIPPO,
        # ---- Behaviour Prior
        "BPIPPO": BehaviourPriorIPPO,
        # -- Tabular
        # ---- Vanilla
        "TabularIA2C": TabularIA2C,
        "TabularMAA2C": TabularMAA2C,
        "TabularIPPO": TabularIPPO,
        "TabularMAPPO": TabularMAPPO,
        # ---- Decoupled
        "TabularDeIPPO": TabularDeIPPO,
        # ---- DoE
        "DoETabularIA2C": DoETabularIA2C,
        "DoETabularIPPO": DoETabularIPPO,
        "DoETabularDeIPPO": DoETabularDeIPPO,
        # ---- Behaviour Prior
        "BPTabularIPPO": BehaviourPriorTabularIPPO,
        "DoEBPIPPO": DoEBPIPPO,
        "DoEBPTabularIPPO": DoEBPTabularIPPO,
        }
    cls = registry[type]
    if not hasattr(cls, "from_config"):
        raise NotImplementedError(f"There is no from_config method defined for {type}")
    else:
        return cls.from_config(cfg, env)
