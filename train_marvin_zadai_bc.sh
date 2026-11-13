#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=${WORKSPACE_ROOT:-/home/xlb/code/强化微调ACT}
RL100_ROOT=${RL100_ROOT:-/home/xlb/code/RL-100/RL-100}
PYTHON_BIN=${PYTHON_BIN:-/home/xlb/miniconda3/envs/rl100/bin/python}

DATASET_PATH=${DATASET_PATH:-/home/xlb/data/zadai_335_cleaned.zarr}
OUTPUT_DIR=${OUTPUT_DIR:-/home/xlb/data/model/marvin_zadai_bc_seed1000}
RESUME=${RESUME:-false}
EPOCHS=${EPOCHS:-150}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-8}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-null}
MAX_VAL_STEPS=${MAX_VAL_STEPS:-null}
SAMPLE_EVERY=${SAMPLE_EVERY:-5}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-10}
SEQUENCE_STRIDE=${SEQUENCE_STRIDE:-32}
N_OBS_STEPS=${N_OBS_STEPS:-2}
N_ACTION_STEPS=${N_ACTION_STEPS:-16}
HORIZON=${HORIZON:-$((N_OBS_STEPS + N_ACTION_STEPS - 1))}

if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi

cd "${RL100_ROOT}"
export PYTHONPATH="${RL100_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-online}"
export TORCH_HOME="${TORCH_HOME:-${WORKSPACE_ROOT}/.cache/torch}"

"${PYTHON_BIN}" train_real.py --config-name=rl100_2d_epsilon.yaml \
  task=marvin_zadai \
  hydra.run.dir="${OUTPUT_DIR}" \
  task.dataset.zarr_path="${DATASET_PATH}" \
  task.dataset.sequence_stride="${SEQUENCE_STRIDE}" \
  training.device=cuda:0 \
  training.seed=1000 \
  training.resume="${RESUME}" \
  training.num_epochs="${EPOCHS}" \
  training.rollout_every=1000000 \
  training.checkpoint_every="${CHECKPOINT_EVERY}" \
  training.val_every=1 \
  training.sample_every="${SAMPLE_EVERY}" \
  training.max_train_steps="${MAX_TRAIN_STEPS}" \
  training.max_val_steps="${MAX_VAL_STEPS}" \
  checkpoint.save_ckpt=true \
  checkpoint.save_last_ckpt=true \
  checkpoint.topk.monitor_key=val_loss \
  checkpoint.topk.mode=min \
  use_wandb=true \
  logging.mode="${WANDB_MODE}" \
  logging.project=marvin-rl100-zadai-bc \
  logging.name="marvin_zadai_seed1000_chunk${N_ACTION_STEPS}" \
  env_num=1 \
  only_bc=true \
  offline=false \
  online=false \
  distill_phase=null \
  feature_type=2D \
  with_pointcloud=false \
  use_agent_pos=true \
  use_action_embed=false \
  action_norm=true \
  chunk_as_single_action=false \
  horizon="${HORIZON}" \
  n_obs_steps="${N_OBS_STEPS}" \
  n_action_steps="${N_ACTION_STEPS}" \
  num_inference_steps=10 \
  policy._target_=rl_100.policy.rl100_2d.RL1002D \
  policy.scheduler_type=ddim \
  policy.model=dp3 \
  policy.w_pc=false \
  policy.use_visual=true \
  policy.use_aug=true \
  policy.img_shape='[3,480,640]' \
  policy.down_dims='[128,256,512]' \
  encoder_type=resnet \
  encoders.resnet.resize_shape='[480,640]' \
  encoders.resnet.crop_shape=null \
  encoders.resnet.random_crop=false \
  encoders.resnet.share_rgb_model=true \
  encoders.resnet.rgb_model.weights=IMAGENET1K_V1 \
  encoders.resnet.imagenet_norm=true \
  use_vib=false \
  use_recon=false \
  kl_annealing=false \
  dataloader.batch_size="${BATCH_SIZE}" \
  dataloader.num_workers="${NUM_WORKERS}" \
  dataloader.persistent_workers=true \
  val_dataloader.batch_size="${BATCH_SIZE}" \
  val_dataloader.num_workers="${NUM_WORKERS}" \
  val_dataloader.persistent_workers=true
