from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy, postprocess_ppo_gae


def my_postprocess_ppo_gae(policy, sample_batch, *args, **kwargs):
    if sample_batch.get('infos') is not None:
        idx = [i for i, x in enumerate(sample_batch['infos']) if x['done']]
        if idx:
            idx.append(sample_batch.count)
            sbatch = sample_batch.slice(0, idx[0] + 1)
            sbatch['dones'][-1] = True
            batch = postprocess_ppo_gae(policy, sbatch, *args, **kwargs)
            for s, t in zip(idx[:-1], idx[1:]):
                sbatch = sample_batch.slice(s, t + 1)
                sbatch['dones'][-1] = True
                batch.concat(
                    postprocess_ppo_gae(policy, sbatch, *args, **kwargs))
            return batch
    return postprocess_ppo_gae(policy, sample_batch, *args, **kwargs)


MyPpoPolicy = PPOTFPolicy.with_updates(name="MyPpoTFPolicy",
                                       postprocess_fn=my_postprocess_ppo_gae)

MyPpoTrainer = PPOTrainer.with_updates(name="MyPpoTrainer",
                                       default_policy=MyPpoPolicy)
