from redis import Redis
from rq import Queue

from maniskill_elirobots.trainer.ppo_cleanrl import main
from maniskill_elirobots.utils import CliArgs

TOTAL_SEEDS = 48
# TOTAL_SEEDS = 2

args_list = [CliArgs(exp_name=f"flipcoin-ec63-{seed}", ent_coef=1e-2, env_id="FlipCoin-v1", seed=seed) for seed in range(TOTAL_SEEDS)]

conn = Redis(host="172.17.0.3")
fila = Queue(name="train", connection=conn)

jobs = [
    fila.enqueue(
        main,
        args,
        job_timeout="10m",
        result_ttl=86400,
        failure_ttl=86400,
    )
    for args in args_list
]

print(jobs)
