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
export LOQUITO_RECORDING_DIR=${HOME}/workspace/data/recordings
export LOQUITO_MODEL_PATH=${HOME}/workspace/data/models/loquito_v1m_hd/2025-04-04_23-22-30/model_epoch_3_torchscript.pt
export LOQUITO_CUDA_DEVICE="cuda:2"

python leaderboard_evaluator_local.py \
--routes ${ROUTES} \
--scenarios ${SCENARIOS} \
--repetitions ${REPETITIONS} \
--track ${CHALLENGE_TRACK_CODENAME} \
--agent ${TEAM_AGENT} \
--debug 0 \
--port 17131
