#!/bin/bash

EXP_NAME=$1

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/..")

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export PYTHONPATH=$PROJECT_DIR
export PYTHONPATH=$PYTHONPATH:$HOME/offline_c_learning/bullet-manipulation
export PYTHONPATH=$PYTHONPATH:$HOME/offline_c_learning/bullet-manipulation/roboverse/envs/assets/bullet-objects
export PYTHONPATH=$PYTHONPATH:$HOME/offline_c_learning/multiworld
export PYTHONPATH=$PYTHONPATH:$HOME/offline_c_learning/railrl-private
export PATH=$PATH:$HOME/anaconda3/envs/railrl/bin
export LOG_ROOT="/projects/rsalakhugroup/chongyiz/offline_c_learning/GoFAR/saved_models/SawyerEnv6/SawyerEnv6-0.1-0.0-gofar-20disc0.01-relabel0.0-0"

declare -a seeds=(0 1)

for seed in "${seeds[@]}"; do
  export CUDA_VISIBLE_DEVICES=$seed
  rm -r "$LOG_ROOT"/run"$seed"
  mkdir -p "$LOG_ROOT"/run"$seed"
  nohup \
  python $PROJECT_DIR/train.py \
    --env SawyerEnv6 \
    --method wgcsl \
    --random_percent 0.0 \
    --n-epochs 300 \
    --n-cycles 50 \
    --presampled_goal_dir /projects/rsalakhugroup/chongyiz/offline_c_learning/dataset/env6_td_pnp_push_1m/goals_early_stop \
    --buffer-size 500_000 \
    --threshold 0.1 \
    --seed "$seed" \
  > "$LOG_ROOT"/run"$seed"/stream.log 2>&1 & \
  sleep 2
done
