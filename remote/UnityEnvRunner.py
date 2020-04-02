import sys
from subprocess import Popen

class UnityEnvRunner:
    def __init__(self):
        self._process, self._port = [], []

    def start(self, port):
        port = port if isinstance(port, list) else [port]
        for p in port:
            if p not in self._port:
                print('Server start UnityEnv on port', p)
                self._port.append(p)
                self._process.append(Popen([sys.executable, 'UnityEnvServer.py', str(p)]))
    
    def restart(self, port):
        self.terminate(port)
        self.run(port)

    def terminate(self, port=None):
        if port is None:
            for process in self._process:
                process.terminate()
            self._process, self._port = [], []
        else:
            port = port if isinstance(port, list) else [port]
            for p in port:
                if p in self._port:
                    print('Server terminate UnityEnv port', p)
                    idx = self._port.index(p)
                    self._port.pop(idx)
                    process = self._process.pop(idx)
                    process.terminate()

import zerorpc
port = 5566 if len(sys.argv) == 1 else int(sys.argv[1])
runner = UnityEnvRunner()
server = zerorpc.Server(runner)
server.bind(f"tcp://0.0.0.0:{port}")
ssh_tunnel = Popen(['ssh', '-f', '-N', '-F', 'config', 'server', '-R', f'{port}:127.0.0.1:{port}'])
try:
    print('Server run on port', port)
    server.run()
finally:
    print('Server release port', port, flush=True)
    server.close()
    ssh_tunnel.terminate()