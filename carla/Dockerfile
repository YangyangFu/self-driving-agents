
# reference: 
#    https://github.com/carla-simulator/leaderboard-cloud/blob/main/leaderboard_2.0/docker/carla/Dockerfile

FROM nvidia/vulkan:1.2.170-470

RUN apt-key adv --fetch-keys "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub"

RUN packages='libsdl2-2.0 xserver-xorg libvulkan1 libomp5' && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y $packages --no-install-recommends

RUN useradd -m carla

COPY --chown=carla:carla . /home/carla

USER carla
WORKDIR /home/carla

ENV OMP_PROC_BIND="FALSE"
ENV OMP_NUM_THREADS="48"
ENV SDL_VIDEODRIVER="x11"
ENV NVIDIA_DRIVER_CAPABILITIES="all"

# you can also run CARLA in offscreen mode with -RenderOffScreen
# CMD /bin/bash CarlaUE4.sh -RenderOffScreen
CMD /bin/bash CarlaUE4.sh
