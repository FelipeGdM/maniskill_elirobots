#!/bin/env fish

set -e DISPLAY

git commit -am "Train commit"

python src/maniskill_elirobots/trainer/ppo_cleanrl.py \
    --ent_coef=1e-2 \
    --exp_name=flipcoin-ec63-(date +%s)-(git rev-parse --short HEAD)-gamma-090 \
    --env_id="FlipCoin-v1" \
    --gamma=0.9

python src/maniskill_elirobots/trainer/ppo_cleanrl.py \
    --ent_coef=1e-2 \
    --exp_name=flipcoin-ec63-(date +%s)-(git rev-parse --short HEAD)-gamma-095 \
    --env_id="FlipCoin-v1" \
    --gamma=0.95

python src/maniskill_elirobots/trainer/ppo_cleanrl.py \
    --ent_coef=1e-2 \
    --exp_name=flipcoin-ec63-(date +%s)-(git rev-parse --short HEAD)-gamma-099 \
    --env_id="FlipCoin-v1" \
    --gamma=0.99

# python vendor/ManiSkill/examples/baselines/ppo/ppo.py \
#     --checkpoint=runs/flipcoin-ec63-1784048581-5f55005/ckpt_8192000.pt \
#     --evaluate \
#     --num-eval-envs=1 \
#     --env-id="maniskill_elirobots:FlipCoin-v1"
