#!/bin/env fish

set -e DISPLAY

# git commit -am "Train commit"

# python src/maniskill_elirobots/trainer/ppo_cleanrl.py \
#     --ent_coef=1e-2 \
#     --exp_name=flipcoin-ec63-(date +%s)-(git rev-parse --short HEAD) \
#     --env_id="FlipCoin-v1"



# python src/maniskill_elirobots/trainer/ppo_cleanrl.py \
python vendor/ManiSkill/examples/baselines/ppo/ppo.py \
    --checkpoint=runs16/FlipCoin-v1__ppo_cleanrl__1__1778612947/ckpt_8192000.pt \
    --evaluate \
    --env-id="maniskill_elirobots:FlipCoin-v1"
