from gym.vector.utils import concatenate, create_empty_array
import supersuit as ss


class AgentDictConcatVecEnv(ss.vector.ConcatVecEnv):
    def __init__(self, par_cvenv, agent_ids, new_step_api=True):
        self.par_cvenv = par_cvenv
        self.vec_envs = self.par_cvenv.vec_envs
        self.metadata = self.par_cvenv.metadata
        self.observation_space = self.par_cvenv.observation_space
        self.action_space = self.par_cvenv.action_space
        raw_agent_ids = self.vec_envs[0].par_env.possible_agents
        self.raw_agent_ids = raw_agent_ids
        self.agent_ids = agent_ids
        self.num_agents = len(raw_agent_ids)
        self.agent_id_to_raw = dict(zip(agent_ids, raw_agent_ids))
        self.agent_raw_to_id = dict(zip(raw_agent_ids, agent_ids))
        self.num_vec_envs = len(par_cvenv.vec_envs)
        self.num_envs = par_cvenv.num_envs
        self.new_step_api = new_step_api

    def step(self, actions):
        raw_actions = self.concatenate_actions(
                [actions[self.agent_raw_to_id[raw_agent]].flatten()[venv_idx]
                 for venv_idx in range(self.num_vec_envs)
                 for raw_agent in self.vec_envs[venv_idx].par_env.possible_agents
                 ],
                n_actions=self.num_envs
                )
        raw_observations, raw_rewards, raw_terminations, raw_truncations, raw_infos = super().step(raw_actions)
        observations = self.split_vec_into_agent_dict(raw_observations)
        rewards = self.split_vec_into_agent_dict(raw_rewards)
        terminations = self.split_vec_into_agent_dict(raw_terminations)
        truncations = self.split_vec_into_agent_dict(raw_truncations)
        infos = self.split_vec_into_agent_dict(raw_infos)
        if not self.new_step_api:
            dones = {agent_id: terminations[agent_id] | truncations[agent_id]
                     for agent_id in terminations.keys()}
            return observations, rewards, dones, infos
        return observations, rewards, terminations, truncations, infos

    def reset(self, seed=None, return_info=False, options=None):
        raw_observations = super().reset(seed=seed, return_info=return_info, options=options)
        observations = self.split_vec_into_agent_dict(raw_observations)
        return observations

    def split_vec_into_agent_dict(self, vec):
        return {self.agent_raw_to_id[raw_agent]: vec[idx::self.num_agents]
                for idx, raw_agent in enumerate(self.raw_agent_ids)
               }


class AgentDictProcConcatVecEnv(ss.vector.ProcConcatVec):
    def __init__(self, par_cvenv, agent_ids, raw_agent_ids, num_vec_envs, new_step_api=True):
        self.par_cvenv = par_cvenv
        self.metadata = self.par_cvenv.metadata
        self.observation_space = self.par_cvenv.observation_space
        self.action_space = self.par_cvenv.action_space
        self.raw_agent_ids = raw_agent_ids
        self.agent_ids = agent_ids
        self.num_agents = len(raw_agent_ids)
        self.agent_id_to_raw = dict(zip(agent_ids, raw_agent_ids))
        self.agent_raw_to_id = dict(zip(raw_agent_ids, agent_ids))
        self.num_vec_envs = num_vec_envs
        self.num_envs = self.par_cvenv.num_envs

        self.idx_starts = self.par_cvenv.idx_starts
        self.pipes = self.par_cvenv.pipes
        self.procs = self.par_cvenv.procs
        self.spec = self.par_cvenv.spec
        self.observations_buffers = self.par_cvenv.observations_buffers
        self.call = self.par_cvenv.call
        self.call_wait = self.par_cvenv.call_wait
        self.call_async = self.par_cvenv.call_async
        self.reset_wait = self.par_cvenv.reset_wait
        self.reset_async = self.par_cvenv.reset_async
        self.step_async = self.par_cvenv.step_async
        self.step_wait = self.par_cvenv.step_wait
        self.close = self.par_cvenv.close
        self.close_extras = self.par_cvenv.close_extras
        self.new_step_api = new_step_api

    def step(self, actions):
        raw_actions = concatenate(
            self.action_space,
            [actions[self.agent_raw_to_id[raw_agent]].flatten()[venv_idx]
             for venv_idx in range(self.num_vec_envs)
             for raw_agent in self.raw_agent_ids
             ],
            create_empty_array(self.action_space, n=self.num_envs)
            )
        raw_observations, raw_rewards, raw_terminations, raw_truncations, raw_infos = super().step(raw_actions)
        observations = self.split_vec_into_agent_dict(raw_observations)
        rewards = self.split_vec_into_agent_dict(raw_rewards)
        terminations = self.split_vec_into_agent_dict(raw_terminations)
        truncations = self.split_vec_into_agent_dict(raw_truncations)
        infos = self.split_vec_into_agent_dict(raw_infos)
        if not self.new_step_api:
            dones = {agent_id: terminations[agent_id] | truncations[agent_id]
                     for agent_id in terminations.keys()}
            return observations, rewards, dones, infos
        return observations, rewards, terminations, truncations, infos

    def reset(self, seed=None, return_info=False, options=None):
        raw_observations = super().reset(seed=seed, return_info=return_info, options=options)
        observations = self.split_vec_into_agent_dict(raw_observations)
        return observations

    def split_vec_into_agent_dict(self, vec):
        return {self.agent_raw_to_id[raw_agent]: vec[idx::self.num_agents]
                for idx, raw_agent in enumerate(self.raw_agent_ids)
               }

def agent_dict_concat_vec_env_v0(cvenv, agent_ids, **kwargs):
    """
    This wrapper modifies the inputs and outputs of the ConcatVecEnv class to use an 'agent dict'
    format.

    The ConcatVecEnv class process actions/observations in the form
        [env_0_agent_0_val, env_0_agent_1_val, ..., env_0_agent_N_val, env_1_agent_0_val, ...]

    This wrapper changes the action/observation(/etc) representation to an agent dict, i.e.:
        {"agent_0_id": [env_0_agent_0_val, env_1_agent_0_val, ..., env_M_agent_0_val],
         "agent_1_id": [env_0_agent_1_val, env_1_agent_1_val, ..., env_M_agent_1_val],
         ...,
         "agent_N_id": [env_0_agent_N_val, env_1_agent_N_val, ..., env_M_agent_N_val],
         }
    """
    if kwargs.get("num_cpus", 0) > 1:
        return AgentDictProcConcatVecEnv(
            cvenv,
            agent_ids,
            raw_agent_ids=kwargs["raw_agent_ids"],
            num_vec_envs=kwargs["num_vec_envs"],
            new_step_api=kwargs.get("new_step_api", False))
    else:
        return AgentDictConcatVecEnv(
            cvenv,
            agent_ids,
            new_step_api=kwargs.get("new_step_api", False)
            )
