# get current project path by using the file path
# use the directory of this file as the project path
export PROJECT_PATH=$(cd `dirname $BASH_SOURCE`; pwd)
echo ${PROJECT_PATH}

# carla version
export CARLA_VERSION=0.9.14

# set up paths
export LEADERBOARD_ROOT=${PROJECT_PATH}/leaderboard
export SCENARIO_RUNNER_ROOT=${PROJECT_PATH}/scenario_runner
export PYTHONPATH=${PROJECT_PATH}:${LEADERBOARD_ROOT}:${SCENARIO_RUNNER_ROOT}:${PYTHONPATH}
