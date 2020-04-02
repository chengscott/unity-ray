from env import UnityEnv, UnityEnvRunner
from model import PpoModel
from policy import MyPpoTrainer, MyPpoPolicy
from metric import my_loggers, metric_callbacks
import ray
from ray import tune
from ray.rllib.models import ModelCatalog

if __name__ == "__main__":
    runner = UnityEnvRunner(control_port=10006,
                            server_port=17000,
                            num_envs=[1, 2, 1])
    env_config = dict(
        environment_path=
        #r'C:\Users\cgilab\Documents\chengscott\bin\v20200303-attack4\G310.exe',
        r'C:\Users\Administrator\Documents\chengscott\bin\v20200303-attack4\G310.exe',
        port=20000,
        use_visual=False,
        use_vector=True,
        multiagent=True,
        uint8_visual=False,
        flatten_branched=False,
        server_port=runner.server_port,
    )
    if True:
        runner.start_test()
        init_env = UnityEnv(env_config)
        act_space, obs_space = init_env.action_space, init_env.observation_space
        init_env.close()
        runner.terminate_test()

    runner.start()
    ray.init()  #local_mode=True)
    ModelCatalog.register_custom_model("ppo_model", PpoModel)
    try:
        tune.run(
            MyPpoTrainer,
            loggers=my_loggers,
            stop={
                "timesteps_total": 80000000,
            },
            config={
                # === Settings for Rollout Worker processes ===
                "num_workers": runner.num_workers,
                "batch_mode": "truncate_episodes",
                # === Settings for the Trainer process ===
                "num_gpus": 1,
                "num_cpus_per_worker": 1,
                "num_gpus_per_worker": 0,
                # === Environment Settings ===
                "env": UnityEnv,
                "env_config": env_config,
                "horizon": 5000,
                "soft_horizon": True,
                "no_done_at_end": True,
                "lr_schedule": [[0, 0.00025], [80000000, 0.0]],
                # === PPO-specific Settings ===
                "gamma": 0.998,
                "lambda": 0.95,
                "kl_coeff": 0.0,
                "vf_loss_coeff": 1.0,
                "entropy_coeff_schedule": [[0, 0.01], [80000000, 0.0]],
                "clip_param": 0.2,
                "vf_clip_param": 10.0,
                "grad_clip": 0.5,
                "sample_batch_size": 400,  # rollout_fragment_length
                "train_batch_size": 1200,
                "sgd_minibatch_size": 16,
                "num_sgd_iter": 4,
                # === Model Settings ===
                "fcnet_hiddens": [256, 256],
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
                # === Debug Settings ===
                "monitor": True,
                "log_level": "INFO",
                "callbacks": metric_callbacks,
            },
            checkpoint_freq=10,
            max_failures=5,
        )
    finally:
        runner.terminate()
