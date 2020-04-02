import pickle
import zlib
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import zerorpc
import msgpack_numpy as m
m.patch()


class UnityEnvRunner:
    def __init__(self, **config):
        port = config['control_port']
        self.server_port = server_port = config['server_port']
        num_envs = config['num_envs']
        self._num_workers = sum(num_envs)
        num_envs[0] += 1  # extra worker
        n_clients = len(num_envs)
        # server ports
        base = 0
        self._server_port = []
        for n in num_envs:
            self._server_port.append(
                [server_port + x + base for x in range(n)])
            base += n
        # connect clients
        self._client = [zerorpc.Client(heartbeat=30) for _ in range(n_clients)]
        for i in range(n_clients):
            self._client[i].connect(f'tcp://127.0.0.1:{port + i}')

    def start_test(self):
        self._client[0].start(self._server_port[0][0])

    def start(self):
        for client, ports in zip(self._client, self._server_port):
            client.start(ports)

    def restart(self):
        for client, ports in zip(self._client, self._server_port):
            client.restart(ports)

    def terminate_test(self):
        self._client[0].terminate(self._server_port[0][0])

    def terminate(self):
        for client, ports in zip(self._client, self._server_port):
            client.terminate(ports)

    def close(self):
        for client in self._client:
            client.close()

    @property
    def num_workers(self):
        return self._num_workers


class UnityEnv(MultiAgentEnv):
    def __init__(self, env_config):
        idx = getattr(env_config, 'worker_index', 0)
        self._client = zerorpc.Client(heartbeat=30)
        ip = env_config.get('server_ip', '127.0.0.1')
        port = env_config['server_port'] + idx
        env_config['port'] += idx
        print('connect', ip, port)
        self._client.connect(f'tcp://{ip}:{port}')
        self._client.launch(env_config)
        self._client.reset()
        # setup env spaces
        self.action_space, self.observation_space, self.agent_name = self._get_env_spaces(
        )

    def step(self, action):
        return self._client.step(action)

    def reset(self):
        return self._client.reset()

    def close(self):
        self._client.env_close()
        self._client.close()

    def _get_env_spaces(self):
        z = self._client.get_env_spaces()
        p = zlib.decompress(z)
        return pickle.loads(p)

    def get_action_count(self):
        return self._client.get_action_count()

    def sample(self):
        return self._client.sample()

    def number_agents(self):
        return self._client.number_agents()


if __name__ == '__main__':
    env_config = dict(
        environment_path=
        r'C:\Users\cgilab\Documents\chengscott\bin\v20200303-attack4\G310.exe',
        port=10000,
        use_visual=False,
        use_vector=True,
        multiagent=True,
        uint8_visual=False,
        flatten_branched=False,
        #server_ip='10.5.11.238',
        server_port=5566,
    )

    env = UnityEnv(env_config)
    obs, reward, done, info = env.reset()
    obs = env.reset()
    n_agents = env.number_agents()
    action = env.sample()
    print(env.action_space)
    print(env.observation_space)
    print(n_agents)
    env.close()
