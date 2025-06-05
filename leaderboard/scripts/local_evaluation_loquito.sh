export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
export BENCHMARK=longest6
export DIRECT=0
export ROUTES=${WORK_DIR}/leaderboard/data/longest6.xml
export SCENARIOS=${WORK_DIR}/leaderboard/data/scenarios/eval_scenarios.json
export REPETITIONS=1
export TEAM_AGENT=${WORK_DIR}/team_code_loquito/loquito_agent.py
export DATAGEN=0
export CHALLENGE_TRACK_CODENAME=SENSORS
export CARLA_PORT=17131

export LOQUITO_LIVE_RESULTS=1
export LOQUITO_RECORDING_DIR=/work/data01/ippen/data/recordings6

python leaderboard_evaluator_local.py \
--routes ${ROUTES} \
--scenarios ${SCENARIOS} \
--repetitions ${REPETITIONS} \
--track ${CHALLENGE_TRACK_CODENAME} \
--agent ${TEAM_AGENT} \
--debug 0 \
--port 17131
