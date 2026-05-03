#!/bin/env fish

set -e DISPLAY

python src/maniskill_elirobots/trainer/ppo_cleanrl.py \
    --qvel_tolerance=0.5 \
    --qvel_penalty=0.2 \
    --exp_name="pickcube-ec63-18"


qvel_tolerance - 0.25 0.5 1.0
qvel_penalty - 0.125 0.25 0.5 1.0

python src/maniskill_elirobots/trainer/ppo_cleanrl.py --qvel_tolerance=0.5 --qvel_penalty=0.2 --exp_name="pickcube-ec63"

python src/maniskill_elirobots/trainer/ppo_cleanrl.py --qvel_tolerance={qt} --qvel_penalty={qp} --exp_name="pickcube-ec63-{qt}-{qp}"
