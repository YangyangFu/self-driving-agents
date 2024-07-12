# End-to-end Self-driving Agents
Goal:
- implement various end-to-end self-driving agents 
- perform closed-loop simulation based on Carla simulator

## Scope
The following self-diving agents are being implemented:
- [ ] Interfuser
- [ ] Transfuser
- [ ] MTR/MTR++
- [ ] CarLLaVA
- [ ] GAIL


### Run Carla in Docker

The followings steps are for running Carla 0.9.14 in a docker container, and connect to it using Python client in local.
- open a terminal, and run one server using the scripts in `./run_server`.
  ```bash
    bash ./run_server/run_carla_0.9.14.sh
  ``` 
- open another terminal to run a client code in local
    ```bash
    conda activate carla-py37
    source setup.sh
    python run_agent.py
    ```
### Run Leaderboard 2.0
The following steps are for runing Carla Leaderboard 2.0 examples in local:
- open a terminal, and run one server using the scripts in `./run_server`.
  ```bash
    source setup_leaderboard2.0.sh
    bash ./run_server/run_carla_leaderboard2.0.sh
  ``` 
- open another terminal to run the client code in local
    - activate local conda environment, e.g., 
        ```bash
        conda activate carla-py37
        ```
    - set up necessary environment variables
        ```bash
        source setup_leaderboard2.0.sh
        ```
    - run a human agent example using
        ```bash
        bash run_agent.sh
        ```
- open a third terminal to run manual control
    ```bash
    conda activate carla-py37
    python ./scenario_runner/manual_control.py
    ```
## Carla in Docker

### Ubuntu 20.04
If we want to run Carla in Docker, and `render the GUI` on the display in the host machine:
- `0.9.10.1`: 
- `0.9.12`: the command on the homepage as follows, does not work
    ```
    sudo docker run --privileged --gpus all --net=host -e DISPLAY=$DISPLAY carlasim/carla:0.9.12 /bin/bash ./CarlaUE4.sh
    ```
    This is no error message, but just not running.

If we simply use off-screen mode,
- `0.9.10.1`:
- `0.9.12`: the following command from homepage will work
    ```
    sudo docker run --privileged --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:0.9.12 /bin/bash ./CarlaUE4.sh -RenderOffScreen
    ```

### Windows

Carla on Windows SUCKS. 

-`0.9.14`: The shipped binaries from github release cannot load `Town06/06 and Town12` and maybe others. 
It seems one has to build Carla from source to get it working according to [this](https://github.com/carla-simulator/carla/issues/6616).


## SETUP
- if needed, `carla_birdeye_view` library is outdated.
  - the map name in Carla 0.9.1 has changed to `Carla/Maps/TownXX/TownXX` instead of `TownXX` (as programmed). The map name in `parametrized_cache_path()` should be updated so that this can be taken care of.

