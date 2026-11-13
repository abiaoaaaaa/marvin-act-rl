#!/usr/bin/env bash
set -euo pipefail

# A new 100-action architecture. It must not resume the older 16-action run.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -d "${SCRIPT_DIR}/../RL-100/rl_100" ]]; then
  export WORKSPACE_ROOT="${WORKSPACE_ROOT:-${SCRIPT_DIR}}"
  export RL100_ROOT="${RL100_ROOT:-${SCRIPT_DIR}/../RL-100}"
  export PYTHON_BIN="${PYTHON_BIN:-/home/zzx23457/miniconda3/envs/rl100/bin/python}"
  if [[ -z "${OUTPUT_DIR:-}" ]]; then
    echo "Remote chunk100 training requires explicit OUTPUT_DIR on a disk with at least 600 GB free." >&2
    exit 2
  fi
fi
export OUTPUT_DIR="${OUTPUT_DIR:-/home/xlb/data/model/marvin_zadai_bc_chunk100_seed1000}"
if [[ "${RESUME:-false}" == "true" && "${OUTPUT_DIR}" == "/home/xlb/data/model/marvin_zadai_bc_seed1000" ]]; then
  echo "chunk100 cannot resume /home/xlb/data/model/marvin_zadai_bc_seed1000 (it has 16 actions)." >&2
  exit 2
fi
export N_OBS_STEPS=2
export N_ACTION_STEPS=100
export HORIZON=101
export BATCH_SIZE="${BATCH_SIZE:-4}"
exec /home/xlb/code/强化微调ACT/train_marvin_zadai_bc.sh
