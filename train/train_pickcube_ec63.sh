#!/bin/env fish

set -e DISPLAY

python src/maniskill_elirobots/trainer/ppo_cleanrl.py \
    --exp_name="pickcube-ec63-03-limits" 
