from dataclasses import asdict

from maniskill_elirobots.trainer.ppo_cleanrl import main
from maniskill_elirobots.utils import CliArgs
from maniskill_elirobots.workers.celery_app import app

# TOTAL_SEEDS = 48
TOTAL_SEEDS = 3


@app.task(bind=True, time_limit=600)  # pyright: ignore[reportUntypedFunctionDecorator]
def execute_main(args: dict, *largs):
    print(largs)
    return main(CliArgs(args))


args_list = [CliArgs(exp_name=f"flipcoin-ec63-{seed}", ent_coef=1e-2, env_id="FlipCoin-v1", seed=seed) for seed in range(TOTAL_SEEDS)]

jobs = [execute_main.delay(asdict(args)) for args in args_list]  # pyright: ignore[reportFunctionMemberAccess]
