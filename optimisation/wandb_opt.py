from typing import Tuple

import pandas as pd
import wandb
from matplotlib.pylab import f
from wandb.integration.sb3 import WandbCallback

from agents.drl_agent import DRLAgent
from config import config_models
from environments.env_portfolio_optimisation import (
    PortfolioOptimisationEnvWrapper,
)
from pbenchmark.portfolio_benchmark import PortfolioBenchmark

wandb.login()


class WandbOptimisation:
    def __init__(
        self,
        entity: str,
        project: str,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        state_columns: list,
    ):
        """
        Initialize the WANDBOptimisation class.
        :param entity: The entity name for Weights & Biases.
        :param project: The project name for Weights & Biases.
        :param train_data: The training data.
        :param val_data: The validation data.
        :param test_data: The test data.
        :param state_columns: The columns that represent the environment.
        """
        self.entity = entity
        self.project = project
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.state_columns = state_columns

        # Initialise wandb API
        self.api = wandb.Api()

        # Initialise portfolio benchmark
        self.portfolio_benchmark = PortfolioBenchmark()

    def wandb_train(
        self,
        model_name: str,
    ):
        """
        Train a model using Weights & Biases.
        :param model_name: The name of the model to train.
        :param train_data: The training data.
        :param val_data: The validation data.
        :param state_columns: The columns that represent the environment.
        """
        with wandb.init(settings={"quiet": "True"}) as run:
            environment = PortfolioOptimisationEnvWrapper(
                train_data=self.train_data,
                trade_data=self.val_data,
                state_columns=self.state_columns,
                verbose=0,
            )

            env_train = environment.get_train_env()
            gym_env, _ = environment.get_trade_env()

            agent = DRLAgent(run)

            configuration = wandb.config.as_dict()

            model = agent.get_model(
                model_name,
                model_kwargs=configuration,
                environment=env_train,
                directory=None,
                use_case="portfolio-optimisation",
                verbose=0,
            )

            trained_model = agent.train(
                model,
                tb_log_name=model_name,
                callback=WandbCallback(),
            )

            df_account, _ = agent.predict(trained_model, gym_env)

            metrics = {
                "sharpe_ratio": self.portfolio_benchmark.compute_sharpe_ratio(
                    df_account
                ),
                "cumulative_return": self.portfolio_benchmark.compute_cum_returns(
                    df_account
                ),
            }

            wandb.log(metrics)

    def sweep(
        self,
        sweep_config: dict,
        model_name: str,
        number_trials: int = 5,
    ) -> str:
        """
        Start a sweep for hyperparameter optimization.
        :param sweep_config: The sweep configuration.
        :param model_name: The name of the model to optimize.
        :param number_trials: The number of trials to run.
        :return: The sweep ID.
        """
        configuration = sweep_config
        configuration["parameters"] = config_models.MODEL_SWEEP_CONFIG[
            model_name
        ]

        sweep_id = wandb.sweep(configuration, project=self.project)
        wandb.agent(
            sweep_id,
            lambda model_name=model_name: self.wandb_train(model_name),
            count=number_trials,
        )

        return sweep_id

    def get_best_sweep(
        self, sweep_id: str, model_name: str
    ) -> Tuple[str, dict]:
        """
        Get the best sweep run for a given model.
        :param sweep_id: The ID of the sweep.
        :param model_name: The name of the model.
        :return: The best run ID and configuration.
        """
        sweep = self.api.sweep(f"{self.entity}/{self.project}/{sweep_id}")
        runs = sweep.runs

        best_run = max(
            runs,
            key=lambda run: run.summary.get("sharpe_ratio", 0),
        )
        best_run_config = best_run.config

        best_config = dict()
        print("Best run configuration:")
        for key in config_models.MODEL_SWEEP_CONFIG[model_name]:
            best_config[key] = best_run_config.get(key, 0)
            if type(best_config[key]) is float:
                print(f"\t{key}: {best_config[key]:.6f}")
            else:
                print(f"\t{key}: {best_config[key]}")

        print("Best run metrics:")
        for key in [
            "sharpe_ratio",
            "cumulative_return",
        ]:
            print(f"\t{key}: {best_run.summary.get(key, 0):.4f}")

        return best_run.id, best_config

    def test_best_run(
        self,
        model_name: str,
        configuration: dict,
        train_val_data: pd.DataFrame,
        logs_directory: str,
        models_directory: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Test the best run configuration for a given model.
        :param model_name: The name of the model.
        :param configuration: The configuration of the model.
        :param train_val_data: The training and validation data.
        :param logs_directory: The directory to save logs.
        :param models_directory: The directory to save models.
        :return: The account and actions DataFrames.
        """
        environment = PortfolioOptimisationEnvWrapper(
            train_data=train_val_data,
            trade_data=self.test_data,
            state_columns=self.state_columns,
        )

        env_train = environment.get_train_env()
        gym_env, _ = environment.get_trade_env()

        agent = DRLAgent()

        model = agent.get_model(
            model_name,
            model_kwargs=configuration,
            environment=env_train,
            directory=logs_directory,
            use_case="portfolio-optimisation",
        )

        print(f"Training model: {model_name.upper()}")
        trained_model = agent.train(
            model,
            tb_log_name=model_name,
        )

        print(f"Saving model: {model_name.upper()}")
        agent.save_model(
            model=model,
            model_name=model_name,
            directory=models_directory,
        )

        print(f"Evaluating model: {model_name.upper()}")
        df_account, df_actions = agent.predict(
            model=trained_model,
            environment=gym_env,
        )

        return df_account, df_actions
