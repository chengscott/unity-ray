import numpy as np
from collections import defaultdict
from ray.tune.logger import Logger, DEFAULT_LOGGERS


class MyLogger(Logger):
    metrics = ('Cumulative Reward', 'Score', 'Kill', 'Death', 'Damage',
               'LoseHP', 'Hit', 'IsCloseWall', 'MaxDistance',
               'ContinuousForward', 'RewardClosewall', 'RewardHitwall',
               'RewardDetect', 'RewardDamage', 'RewardStep', 'RewardKill',
               'RewardFinishDistance', 'RewardMoveForward')

    def _init(self):
        from tensorboardX import SummaryWriter
        self._file_writer = SummaryWriter(self.logdir + '/my_log',
                                          flush_secs=30)

    def on_result(self, result):
        step = result['timesteps_total']
        hist_stats = result['hist_stats']
        if hist_stats.get('Score', []):
            for metric in MyLogger.metrics:
                self._file_writer.add_scalar(metric,
                                             np.mean(hist_stats[metric]),
                                             global_step=step)
                self._file_writer.add_histogram(metric + '_hist',
                                                hist_stats[metric],
                                                global_step=step)
                result['hist_stats'][metric] = []
            self._file_writer.flush()

    def flush(self):
        self._file_writer.flush()

    def close(self):
        self._file_writer.close()


def on_episode_start(info):
    if 'info' not in info['episode'].user_data:
        info['episode'].user_data["info"] = defaultdict(list)


def on_episode_step(info):
    episode = info['episode']
    envs = info['env'].envs
    for env in envs:
        for name in env.agent_name:
            info = episode.last_info_for(name)
            if not info:
                continue
            reward = episode._agent_reward_history[name][-1]
            info['Cumulative Reward'] = reward
            episode.user_data["info"][name].append(info)


def on_episode_end(info):
    episode = info['episode']
    for metric in MyLogger.metrics:
        episode.hist_data[metric] = []
    for name, einfos in episode.user_data["info"].items():
        idx = [i for i, x in enumerate(einfos) if x['done']]
        if idx:
            for metric in MyLogger.metrics:
                emetric = [x[metric] for x in einfos]
                for s, t in zip(idx[:-1], idx[1:]):
                    episode.hist_data[metric].append(np.sum(emetric[s:t + 1]))
                episode.user_data["info"][name] = einfos[idx[-1] + 1:]


my_loggers = DEFAULT_LOGGERS + (MyLogger, )

metric_callbacks = {
    "on_episode_start": on_episode_start,
    "on_episode_step": on_episode_step,
    "on_episode_end": on_episode_end,
}
