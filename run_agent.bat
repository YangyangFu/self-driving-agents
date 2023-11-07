REM set up simulation parameters
set ROUTES=%LEADERBOARD_ROOT%\data\routes_devtest.xml
set REPETITIONS=1
set DEBUG_CHALLENGE=1
set TEAM_AGENT=%LEADERBOARD_ROOT%\leaderboard\autoagents\human_agent.py 
set CHECKPOINT_ENDPOINT=%LEADERBOARD_ROOT%\results.json
set CHALLENGE_TRACK_CODEBANE=SENSORS

python %LEADERBOARD_ROOT%\leaderboard\leaderboard_evaluator.py ^
--routes=%ROUTES% ^
--routes-subset=%ROUTES_SUBSET% ^
--repetitions=%REPETITIONS% ^
--track=%CHALLENGE_TRACK_CODENAME% ^
--checkpoint=%CHECKPOINT_ENDPOINT% ^
--debug-checkpoint=%DEBUG_CHECKPOINT_ENDPOINT% ^
--agent=%TEAM_AGENT% ^
--agent-config=%TEAM_CONFIG% ^
--debug=%DEBUG_CHALLENGE% ^
--record=%RECORD_PATH% ^
--resume=%RESUME%