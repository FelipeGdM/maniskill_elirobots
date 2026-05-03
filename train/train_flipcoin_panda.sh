#!/bin/env fish

set -e DISPLAY

python src/maniskill_elirobots/trainer/ppo_cleanrl.py \
    --env_id="FlipCoin-v1" \
    --exp_name="flipcoin-panda-04-fatcoin" \
    --robot-uid="panda" \
    --num-envs=512 \
    --update_epochs=8 \
    --num_minibatches=32 \
    --total_timesteps=8_192_001 \
    --eval-freq=8 \
    --num-steps=50 \
    --num-eval-steps=50 \
    --control-mode="pd_joint_delta_pos"
