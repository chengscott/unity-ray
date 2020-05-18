from model import PpoModel
from policy import MyPpoTrainer, MyPpoPolicy
import argparse
import gym
import numpy as np
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import tensorflow as tf
from tensorflow.python.framework import graph_util


class UnityDummyEnv(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        high = np.array([np.inf] * env_config['vector_obs_size'])
        self.observation_space = gym.spaces.Box(-high, high)
        self.action_space = gym.spaces.MultiDiscrete(env_config['action_size'])

    def reset(self):
        return np.zeros(self.observation_space.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ray-checkpoint-path", required=True)
    parser.add_argument("-o", "--output", default="../model/ppo.pb")
    args = parser.parse_args()

    env_config = {
        "vector_obs_size": 99,
        "action_size": [3, 3, 5, 3, 2],
    }
    init_env = UnityDummyEnv(env_config)
    act_space, obs_space = init_env.action_space, init_env.observation_space

    ray.init()
    ModelCatalog.register_custom_model("ppo_model", PpoModel)
    trainer = MyPpoTrainer(
        config={
            # === Environment Settings ===
            "env": UnityDummyEnv,
            "env_config": env_config,
            # === Model Settings ===
            "vf_share_layers": True,
            # === Settings for Multi-Agent Environments ===
            "multiagent": {
                "policies": {
                    "default_brain": (MyPpoPolicy, obs_space, act_space, {
                        "model": {
                            "custom_model": "ppo_model",
                        },
                    }),
                },
                "policy_mapping_fn": (lambda _: "default_brain"),
            },
        })
    trainer.restore(args.ray_checkpoint_path)
    #trainer.export_policy_checkpoint("tf_ckpt/", "default_brain")
    #trainer.export_policy_model("tf_model/",  default_brain")
    policy = trainer.get_policy("default_brain")
    sess = policy.get_session()
    input_graph_def = policy.model.graph.as_graph_def()
    output_node_names = ["default_brain/one_hot_action"]
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names)
    with tf.compat.v1.gfile.GFile(args.output, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    print(len(output_graph_def.node), 'ops in the final graph.')
