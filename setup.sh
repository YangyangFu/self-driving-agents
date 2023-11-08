# get current project path by using the file path
# get the path of this file
export PROJECT_PATH=$(cd `dirname $0`; pwd)
echo ${PROJECT_PATH}

# carla version
export CARLA_VERSION=0.9.14

# set up paths
export LEADERBOARD_ROOT=${PROJECT_PATH}/leaderboard
set SCENARIO_RUNNER_ROOT=${PROJECT_PATH}/scenario_runner
set PYTHONPATH=${LEADERBOARD_ROOT};${SCENARIO_RUNNER_ROOT};${PYTHONPATH}
