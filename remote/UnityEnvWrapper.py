import numpy as np
import gym
from gym_unity.envs.unity_env import UnityEnv
import zlib
import pickle
import json


class UnityEnvWrapper:
    def __init__(self, env_config=None, use_eval=False, rpc_mode=False):
        self.env = None
        if not rpc_mode:
            assert (env_config is not None)
            self.launch(env_config, use_eval)

    def launch(self, env_config, use_eval=False):
        environment_path = (env_config["environment_path_eval"]
                            if use_eval else env_config["environment_path"])

        port = env_config.get("port", 0)
        if use_eval and port:
            port += 2
        use_visual = env_config.get("use_visual", False)
        use_vector = env_config.get("use_vector", True)
        multiagent = env_config.get("multiagent", False)
        uint8_visual = env_config.get("uint8_visual", True)
        flatten_branched = env_config.get("flatten_branched", True)

        self.env = UnityEnv(
            environment_path,
            port,
            use_visual=use_visual,
            use_vector=use_vector,
            uint8_visual=uint8_visual,
            multiagent=multiagent,
            flatten_branched=flatten_branched,
        )
        self.action_space = self.env._action_space
        self.observation_space = self.env._observation_space
        # agent name must be unique among **all** agents
        self.agent_name = [
            f'{port}_{i}' for i in range(self.env.number_agents)
        ]

    def _transform_list_to_dict(self, objs):
        return {name: obj for name, obj in zip(self.agent_name, objs)}

    def _transform_dict_to_list(self, objs):
        return [objs[name] for name in self.agent_name]

    def step(self, act, action_settings=None):
        action = np.stack(self._transform_dict_to_list(act)).tolist()
        observation, reward, done, info = self.env.step(action)
        transform = self._transform_list_to_dict
        info = list(map(json.loads, info['text_observation']))
        for i, x in enumerate(info):
            x['done'] = done[i]
        done = [False] * 4
        done_dict = transform(done)
        done_dict['__all__'] = False  # no early termination (for logging)
        return transform(observation), transform(reward), done_dict, transform(
            info)

    def reset(self, reset_settings=None):
        obs = self.env.reset()
        return self._transform_list_to_dict(obs)

    def get_env_spaces(self):
        spaces = self.action_space, self.observation_space, self.agent_name
        p = pickle.dumps(spaces)
        z = zlib.compress(p)
        return z

    def get_action_count(self):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return self.env.action_space.n
        elif isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            return self.env.action_space.nvec.tolist()
        raise NotImplementedError

    def sample(self):
        return self.env.action_space.sample()

    def number_agents(self):
        return self.env.number_agents

    def env_close(self):
        if self.env:
            self.env.close()
            self.env = None

    def close(self):
        self.env_close()

    def hello(self):
        print('Hello World')
