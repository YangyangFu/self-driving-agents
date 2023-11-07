# Carla 


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