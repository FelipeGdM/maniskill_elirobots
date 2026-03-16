#!/bin/env fish

# --exp-name="state-pushcube-5" \
    # --total_timesteps=1_200_000 \


python src/maniskill_elirobots/scripts/ppo.py \
    --env_id="PushCubeEc-v1" \
    --num_envs=1024 \
    --update_epochs=8 \
    --num_minibatches=32 \
    --total_timesteps=4_096_001 \
    --eval_freq=8 \
    --num-steps=50 \
    --num-eval-steps=50

    # --checkpoint="runs/PushCubeEc-v1__ppo__1__1773663971/ckpt_5120000.pt" \

    # --checkpoint="runs/PushCubeEc-v1__ppo__1__1773661708/ckpt_1177600.pt" \

    # --total_timesteps=2_048_001 \
