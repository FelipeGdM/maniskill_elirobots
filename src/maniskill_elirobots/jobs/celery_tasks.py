from dataclasses import asdict

from maniskill_elirobots.jobs.celery_app import app
from maniskill_elirobots.trainer.ppo_cleanrl import main
from maniskill_elirobots.utils import CliArgs

TOTAL_SEEDS = 60
# TOTAL_SEEDS = 3


@app.task
def execute_main(args: dict):
    cli_args = CliArgs(**args)
    return main(cli_args)


args_list = [
    CliArgs(
        exp_name=f"flipcoin-ec63-{seed:03d}",
        ent_coef=1e-2,
        env_id="FlipCoin-v1",
        seed=seed,
    )
    for seed in range(100, 100 + TOTAL_SEEDS)
]

if __name__ == "__main__":
    jobs = [execute_main.delay(asdict(args)) for args in args_list]  # pyright: ignore[reportFunctionMemberAccess]
