from UnityEnvWrapper import UnityEnvWrapper
from subprocess import Popen
import sys
import zerorpc
import msgpack_numpy as m
m.patch()

port = 5566 if len(sys.argv) == 1 else int(sys.argv[1])
env = UnityEnvWrapper(rpc_mode=True)
server = zerorpc.Server(env)
server.bind(f"tcp://0.0.0.0:{port}")
ssh_tunnel = Popen(['ssh', '-f', '-N', '-F', 'config', 'server', '-R', f'{port}:127.0.0.1:{port}'])
try:
    print('UnityEnv run on port', port)
    server.run()
finally:
    print('UnityEnv release port', port, flush=True)
    server.close()
    env.close()
    ssh_tunnel.terminate()