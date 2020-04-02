# unity-ray

This package aims to support Ray-RLlib training with Windows-only Unity environments.

- `train/` for Linux Ray-RLlib training
- `remote/` for Windows Unity environments with SSH remote port forwarding

## Run

### Start Unity Environment

One also needs to provide the SSH tunnel server:

```
Host server
HostName tunnel_host
StrictHostKeyChecking no
ServerAliveInterval 10
ControlMaster auto
ControlPersist yes
Compression yes
```

```shell
(venv-unity) win-01$ python UnityEnvRunner.py 10001
(venv-unity) win-02$ python UnityEnvRunner.py 10002
```

`win-01` forwards local port 10001 to remote `server`, and `win-02` forwards local port 10002 to remote `server`.

### Start Training

- `control_port` is used to communicate with `UnityEnvRunner`
- `server_port` is used to communicate with `UnityEnvServer`
- `port` of `UnityEnv` is only used inside localhost

Both `control_port` and `server_port` binds to the ssh-tunnel `server`. (i.e., must not conflict.)

```shell
(venv-ray) linux $ python run.py
```