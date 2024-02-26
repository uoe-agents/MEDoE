# TODO should probably move this to a test folder
import os
import time
from omegaconf import DictConfig
import pettingzoo.test as pzt
import supersuit as ss
import fst.envs.chainball.chainball as chainball
from fst.envs.agent_dict_concat_vecenv import agent_dict_concat_vec_env_v0


cfg = DictConfig({
        "chain_length": 7,
        "spawn_location": 4,
        "optimals": [(1,1), (1,1), (2,1), (2,2), (2,2)],
        "fixed_time": True,
        "max_cycles": 20,
        #"extra_respawn_states": [1,2],
        "enable_our_goal": False,
        "obs_mode": "onehot",
        "render_mode": "human",
        })

env = chainball.env(loader="config", cfg=cfg)
p_env = chainball.parallel_env(loader="config", cfg=cfg)

pzt.max_cycles_test(chainball)
pzt.seed_test(chainball.env)
pzt.api_test(env, num_cycles=1000)
pzt.parallel_api_test(p_env, num_cycles=1000)

p_env = chainball.parallel_env(loader="config", cfg=cfg)

mve = ss.pettingzoo_env_to_vec_env_v1(p_env)
cve = ss.concat_vec_envs_v1(mve, 4)
ade = agent_dict_concat_vec_env_v0(cve, ["bill", "ben"], new_step_api=True)
p_env.reset()
import pdb; pdb.set_trace()
step = 0
os.system("clear")
p_env.render()
print(f"step: {step:>3}   reward: 0")
time.sleep(1)
while p_env.agents:
    os.system("clear")
    action = {agent: p_env.action_space(agent).sample()
             for agent in p_env.agents}
    o,r,te,tr,i = p_env.step(action)
    step += 1
    p_env.render()
    print(f"step: {step:>3}   reward: {r['player_0']}")
    time.sleep(1)
