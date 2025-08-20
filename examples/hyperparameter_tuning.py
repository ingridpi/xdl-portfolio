# Hyper-parameter tuning

import os
import sys

REPO_ROOT = "/Users/ingridperez/Documents/GitHub Repositories/xdl-portfolio"
sys.path.append(REPO_ROOT)

from config import config
from optimisation.wandb_opt import WandbOptimisation
from pbenchmark.portfolio_benchmark import PortfolioBenchmark
from preprocessor.findata_preprocessor import FinancialDataPreprocessor
from visualiser.findata_visualiser import FinancialDataVisualiser
from visualiser.model_visualiser import ModelVisualiser

USE_CASE = "portfolio-optimisation"

data_dir = f"{REPO_ROOT}/{config.DATA_DIR}/{config.DATASET_NAME}"
plot_dir = (
    f"{REPO_ROOT}/{config.PLOT_DIR}/{config.TICKERS_NAME}/{config.DATASET_NAME}"
)
models_dir = f"{REPO_ROOT}/{config.MODELS_DIR}/{USE_CASE}/{config.TICKERS_NAME}/{config.DATASET_NAME}"
results_dir = (
    f"{REPO_ROOT}/{config.RESULTS_DIR}/{USE_CASE}/{config.DATASET_NAME}"
)
logs_dir = f"{REPO_ROOT}/{config.LOGS_DIR}/{USE_CASE}/{config.TICKERS_NAME}/{config.DATASET_NAME}"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

finpreprocessor = FinancialDataPreprocessor(
    start_date=config.START_DATE,
    end_date=config.END_DATE,
)
train_val_data, trade_data = finpreprocessor.load_train_test_data(
    directory=data_dir,
    filename=config.TICKERS_NAME,
)

# Split the training data into training and validation sets
train_data, val_data = finpreprocessor.split_train_test(
    data=train_val_data,
    test_start_date=config.VAL_START_DATE,
)

visualiser = FinancialDataVisualiser(directory=plot_dir)
visualiser.plot_train_val_test_close_prices(
    train_data=train_data,
    val_data=val_data,
    test_data=trade_data,
)

wandb_opt = WandbOptimisation(
    entity=config.WANDB_ENTITY,
    project=config.WANDB_PROJECT,
    train_data=train_data,
    val_data=val_data,
    test_data=trade_data,
    state_columns=config.ENVIRONMENT_COLUMNS,
)

model_visualiser = ModelVisualiser(directory=plot_dir)
benchmark = PortfolioBenchmark()

sweep_ids = {}
best_runs = {}
perf_stats = {}


def perform_model_sweep(model_name: str):

    # Start sweep
    sweep_id = wandb_opt.sweep(
        sweep_config=config.SWEEP_CONFIG,
        model_name=model_name,
    )
    sweep_ids[model_name] = sweep_id

    # Retrieve best run
    run_id, configuration = wandb_opt.get_best_sweep_run(
        sweep_id=sweep_id, model_name=model_name
    )

    best_runs[model_name] = (run_id, configuration)

    # Test the best hyperparameters on the test_set
    df_account, df_actions = wandb_opt.test_best_run(
        model_name=model_name,
        configuration=configuration,
        train_val_data=train_val_data,
        logs_directory=logs_dir,
        models_directory=models_dir,
    )

    model_visualiser.evaluate_testing(
        model_name=model_name,
        account_data=df_account,
        actions_data=df_actions,
    )

    # Compute performance statistics
    perf_stats_alg = benchmark.compute_perf_stats(df_account=df_account)
    perf_stats[model_name] = perf_stats_alg


# A2C
model_name = "a2c"
perform_model_sweep(model_name)

# PPO
model_name = "ppo"
perform_model_sweep(model_name)

# DDPG
model_name = "ddpg"
perform_model_sweep(model_name)

# TD3
model_name = "td3"
perform_model_sweep(model_name)

# SAC
model_name = "sac"
perform_model_sweep(model_name)
