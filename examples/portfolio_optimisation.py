# Portfolio Optimisation Example

import os
import sys

REPO_ROOT = "/Users/ingridperez/Documents/GitHub Repositories/xdl-portfolio"
sys.path.append(REPO_ROOT)

from agents.drl_agent import DRLAgent
from config import config, config_models
from environments.env_portfolio_optimisation import (
    PortfolioOptimisationEnvWrapper,
)
from preprocessor.findata_preprocessor import FinancialDataPreprocessor
from visualiser.model_visualiser import ModelVisualiser

USE_CASE = "portfolio-optimisation"

data_dir = f"{REPO_ROOT}/{config.DATA_DIR}/{config.DATASET_NAME}"
plot_dir = f"{REPO_ROOT}/{config.PLOT_DIR}/{config.TICKERS_NAME}/{config.DATASET_NAME}/{USE_CASE}"
models_dir = f"{REPO_ROOT}/{config.MODELS_DIR}/{USE_CASE}/{config.TICKERS_NAME}/{config.DATASET_NAME}"
logs_dir = f"{REPO_ROOT}/{config.LOGS_DIR}/{USE_CASE}/{config.TICKERS_NAME}/{config.DATASET_NAME}"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

TRAIN = True

finpreprocessor = FinancialDataPreprocessor(
    start_date=config.START_DATE,
    end_date=config.END_DATE,
)
train_data, trade_data = finpreprocessor.load_train_test_data(
    directory=data_dir,
    filename=config.TICKERS_NAME,
)

environment = PortfolioOptimisationEnvWrapper(
    train_data=train_data,
    trade_data=trade_data,
    state_columns=config.ENVIRONMENT_COLUMNS,
)

model_visualiser = ModelVisualiser(directory=plot_dir)

for model_name in config_models.MODELS.keys():

    env_train = environment.get_train_env()
    gym_env, _ = environment.get_trade_env()

    agent = DRLAgent()

    model = agent.get_model(
        model_name=model_name,
        environment=env_train,
        directory=logs_dir,
        use_case=USE_CASE,
    )

    if TRAIN:
        print(f"Training model: {model_name.upper()}")
        trained_model = agent.train(
            model=model,
            tb_log_name=model_name,
        )
        print(f"Saving model: {model_name.upper()}")
        agent.save_model(
            model=model,
            model_name=model_name,
            directory=models_dir,
        )

        visualisation_config = config_models.train_visualisation_config[
            model_name
        ]

        model_visualiser.evaluate_training(
            model_name=model_name,
            x=visualisation_config["x"],
            y=visualisation_config["y"],
            title=visualisation_config["title"],
            logs_dir=logs_dir,
        )

    else:
        print(f"Loading model: {model_name.upper()}")
        trained_model = agent.load_model(
            model_name=model_name,
            directory=models_dir,
        )

    print(f"Evaluating model: {model_name.upper()}")
    df_account, df_actions = agent.predict(
        model=trained_model,
        environment=gym_env,
    )

    model_visualiser.evaluate_testing(
        model_name=model_name,
        account_data=df_account,
        actions_data=df_actions,
    )
