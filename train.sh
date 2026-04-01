#!/bin/env fish

# --exp-name="state-pushcube-5" \
    # --total_timesteps=1_200_000 \

set -e DISPLAY

python src/maniskill_elirobots/trainer/ppo_cleanrl.py \
    --env_id="PushCubeEc-v1" \
    --exp-name="pushcube-delta-pos-controller-09" \
    --num_envs=1024 \
    --update_epochs=8 \
    --num_minibatches=32 \
    --total_timesteps=8_192_001 \
    --eval_freq=8 \
    --num-steps=50 \
    --num-eval-steps=50 \
    --control-mode="pd_joint_delta_pos"

    # --checkpoint="runs/PushCubeEc-v1__ppo__1__1773663971/ckpt_5120000.pt" \

    # --checkpoint="runs/PushCubeEc-v1__ppo__1__1773661708/ckpt_1177600.pt" \

    # --total_timesteps=2_048_001 \
