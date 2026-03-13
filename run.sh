#!/bin/env fish

# --exp-name="state-pushcube-5" \
    # --total_timesteps=1_200_000 \


python src/maniskill_elirobots/scripts/ppo.py \
    --env_id="PushCubeEc-v1" \
    --exp-name="state-pushcube-les-goo-4" \
    --num_envs=1024 \
    --update_epochs=8 \
    --num_minibatches=32 \
    --total_timesteps=2_400_000 \
    --eval_freq=8 \
    --num-steps=50 \
    --num-eval-steps=100
