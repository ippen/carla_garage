#!/bin/bash
# Must set CARLA_ROOT
export LOQUITO_RECORDING_DIR=/home/ippen/workspace/data/recordings_b2d
export LOQUITO_LIVE_RESULTS=1

export WORK_DIR=/home/ippen/workspace/projects/end2end-driving-model/loquito/eval/closed_loop/carla_garage/Bench2Drive

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/leaderboard
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/scenario_runner
export PYTHONPATH=$PYTHONPATH:/home/ippen/workspace/projects/end2end-driving-model/loquito
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner

export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=13500
export TM_PORT=14500
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True
export IS_BENCH2DRIVE=True
export PLANNER_TYPE=traj
export GPU_RANK=3

# TCP evaluation
export ROUTES=/home/ippen/workspace/projects/end2end-driving-model/loquito/eval/closed_loop/carla_garage/Bench2Drive/leaderboard/data/bench2drive220.xml
export TEAM_AGENT=/home/ippen/workspace/projects/end2end-driving-model/loquito/eval/closed_loop/carla_garage/Bench2Drive/leaderboard/team_code_loquito/loquito_agent.py
export TEAM_CONFIG=""
export CHECKPOINT_ENDPOINT=results_b2d.json
export SAVE_PATH=/home/ippen/workspace/data/recordings_b2d/save

echo -e "CUDA_VISIBLE_DEVICES=${GPU_RANK} python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py --routes=${ROUTES} --repetitions=${REPETITIONS} --track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT} --agent-config=${TEAM_CONFIG} --debug=${DEBUG_CHALLENGE} --record=${RECORD_PATH} --resume=${RESUME} --port=${PORT} --traffic-manager-port=${TM_PORT} --gpu-rank=${GPU_RANK}"

CUDA_VISIBLE_DEVICES=${GPU_RANK} python "${LEADERBOARD_ROOT}"/leaderboard/leaderboard_evaluator.py --routes="${ROUTES}" --repetitions=${REPETITIONS} --track=${CHALLENGE_TRACK_CODENAME} --checkpoint="${CHECKPOINT_ENDPOINT}" --agent="${TEAM_AGENT}" --agent-config="${TEAM_CONFIG}" --debug=${DEBUG_CHALLENGE} --record="${RECORD_PATH}" --resume=${RESUME} --port="${PORT}" --traffic-manager-port="${TM_PORT}" --gpu-rank="${GPU_RANK}" \
