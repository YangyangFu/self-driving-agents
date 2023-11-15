# get current project path by using the file path
# use the directory of this file as the project path
export PROJECT_PATH=$(cd `dirname $BASH_SOURCE`; pwd)
echo ${PROJECT_PATH}

# carla root
export CARLA_ROOT=${HOME}/CARLA_Leaderboard_20

# set up paths
export LEADERBOARD_ROOT=${PROJECT_PATH}/leaderboard
export SCENARIO_RUNNER_ROOT=${PROJECT_PATH}/scenario_runner
export PYTHONPATH=${PROJECT_PATH}:${LEADERBOARD_ROOT}:${SCENARIO_RUNNER_ROOT}:${PYTHONPATH}
