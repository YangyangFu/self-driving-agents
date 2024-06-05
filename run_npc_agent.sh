#!bin/bash

export CARLA_ROOT=carla
export LEADERBOARD_ROOT=leaderboard
export SCENARIO_RUNNER_ROOT=scenario_runner
export EXPERTS_ROOT=experts
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}:${SCENARIO_RUNNER_ROOT}:${EXPERTS_ROOT}

export CHALLENGE_TRACK_CODENAME=SENSORS
export ROUTES=${LEADERBOARD_ROOT}/data/routes_devtest.xml
export ROUTES_SUBSET=0
export TEAM_AGENT=${LEADERBOARD_ROOT}/leaderboard/autoagents/npc_agent.py #npc_agent.py # agent code
export TEAM_CONFIG= 
export CHECKPOINT_ENDPOINT=_result.json # results file
export DEBUG_CHECKPOINT_ENDPOINT=_debug_result.json # debug results file
export SAVE_PATH=data/eval # path for saving episodes while evaluating
#export RESUME=False 

export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export PORT=2000 # same as the carla server port
export TM_PORT=2500 # port for traffic manager, required when spawning multiple servers/clients

python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--routes-subset=${ROUTES_SUBSET} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--debug-checkpoint=${DEBUG_CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--traffic-manager-port=${TM_PORT}
