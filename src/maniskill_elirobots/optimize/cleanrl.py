from typing import cast

import optuna
from pandas import DataFrame

from maniskill_elirobots.trainer.ppo_cleanrl import main
from maniskill_elirobots.utils import CliArgs


def objective(trial: optuna.Trial):
    # learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, step=1e-4)
    # minibatch_size = trial.suggest_int("minibatch_size", 128, 4096, step=128)
    # gamma = trial.suggest_float("gamma", 0.8, 0.99, step=0.01)
    # gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99, step=0.01)
    # clip_coef = trial.suggest_float("clip_coef", 100e-3, 400e-3, step=10e-3)
    # ent_coef = trial.suggest_float("ent_coef", 1e-5, 1e-2, log=True)
    # update_epochs = trial.suggest_int("update_epochs", 4, 12)

    seed = trial.suggest_int("seed", 1, 120)

    gamma = 0.9
    gae_lambda = 0.95
    minibatch_size = 2048
    learning_rate = 5e-4
    ent_coef = 1e-2
    update_epochs = 10

    args = CliArgs(
        seed=seed,
        capture_video=False,
        total_timesteps=1_024_000 * 8,
        eval_freq=16,
        learning_rate=learning_rate,
        gamma=gamma,
        minibatch_size=minibatch_size,
        gae_lambda=gae_lambda,
        # clip_coef=clip_coef,
        ent_coef=ent_coef,
        update_epochs=update_epochs,
        # exp_name=f"flipcoin-{trial.number:03d}-lr{learning_rate:.3e}-ue{update_epochs:02d}-cc{clip_coef:.3e}-ec{ent_coef:.3e}",
        exp_name=f"flipcoin-s{seed:03d}",
        tensorboard_folder="runs_ppo_seeds_2",
    )

    result = main(args)

    return result["mean_reward_eval_metric"]


study_name = "flipcoin_ppo_seeds_2"  # Unique identifier of the study.
storage_name = f"sqlite:///{study_name}.db"

# Cria um estudo Optuna com amostragem TPE (Bayesiana)
study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True,
    direction="maximize",
    # sampler=optuna.samplers.TPESampler(),
    sampler=optuna.samplers.BruteForceSampler(),
    pruner=optuna.pruners.MedianPruner(),
)

# Executa os trials
study.optimize(objective, n_trials=100)

# # Ou apenas analisar os resultados
# print(f"Melhor valor: {study.best_value}")
# # print(f"Melhores parâmetros: {study.best_params}")

# for key, value in study.best_params.items():
#     print(f"{key:>20} : {value:.5e}")

# df_raw = cast("DataFrame", study.trials_dataframe())

# df = df_raw[df_raw["state"] == "COMPLETE"][df_raw["value"] > 0.5].copy()

# for _, row in cast("DataFrame", df).iterrows():
#     learning_rate = row.loc["params_learning_rate"]
#     minibatch_size = row.loc["params_minibatch_size"]
#     gamma = row.loc["params_gamma"]
#     gae_lambda = row.loc["params_gae_lambda"]
#     clip_coef = row.loc["params_clip_coef"]
#     ent_coef = row.loc["params_ent_coef"]
#     print(f"flipcoin-lr{learning_rate:.3e}-ms{minibatch_size:04d}-g{gamma:.3e}-gl{gae_lambda:.3e}-cc{clip_coef:.3e}-ec{ent_coef:.3e}")

# # flipcoin-lr0.00040345233377220855-ms1280-g0.9096059311498702-gl0.91182438054328-cc0.16330201985627696-ec0.008533679181727961
# # flipcoin-lr0.000276512328568977-ms0768-g0.8943151055073313-gl0.966002810293299-cc0.245080097124798-ec0.0036378465403422095
# # param_cols = [col for col in df.columns if col.startswith("params_")]

# # target_col = "value"

# # df_subset = df[[*param_cols, target_col]].copy()

# # # df_subset = df_subset.dropna(subset=[target_col])

# # # Calcula a matriz de correlação usando Spearman
# # corr_matrix = df_subset.corr(method="pearson")

# # # Extrai a correlação de cada parâmetro com a métrica 'value'
# # # Remove a correlação de 'value' com ela mesma (que é 1.0)
# # correlations_with_target = corr_matrix[target_col].drop(target_col)

# # # Ordena da correlação mais positiva para a mais negativa
# # correlations_sorted = correlations_with_target.sort_values(ascending=False)

# # # Exibe os resultados
# # print("Correlação (Spearman) entre parâmetros e a métrica de otimização:")
# # print(correlations_sorted)
# # print(df.head())
