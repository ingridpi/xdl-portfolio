from typing import Optional

import pandas as pd
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from config import config


class FinRLAgent:
    def __init__(self, train_env: DummyVecEnv):
        """
        Initialises the FinRL agent.
        :param train_env: The training environment for the agent.
        """
        self.train_env = train_env
        self.agent = DRLAgent(env=train_env)

    def get_model(
        self,
        model_name: str,
        model_kwargs: Optional[dict] = None,
    ) -> A2C | PPO | DDPG | TD3 | SAC:
        """
        Returns a DRL model based on the specified model name and parameters.
        :param model_name: The name of the DRL model to be used.
        :param model_kwargs: Additional keyword arguments for the model.
        :return: A DRL model instance.
        """
        if model_name not in config.MODELS:
            raise ValueError(
                f"Model {model_name} is not supported. Choose from {config.MODELS}."
            )

        return self.agent.get_model(
            model_name=model_name, model_kwargs=model_kwargs, verbose=0
        )

    def train_model(
        self,
        model_name: str,
        model_kwargs: Optional[dict] = None,
        total_timesteps: int = 100000,
        logger_dir: Optional[str] = config.RESULTS_DIR,
        logger_outputs: Optional[list] = ["stdout", "csv"],
    ) -> A2C | PPO | DDPG | TD3 | SAC:
        """
        Trains the DRL model.
        :param model_name: The name of the DRL model to be trained.
        :param model_kwargs: Additional keyword arguments for the model.
        :param total_timesteps: Total number of timesteps for training.
        :return: The trained DRL model.
        """
        if model_name not in config.MODELS:
            raise ValueError(
                f"Model {model_name} is not supported. Choose from {config.MODELS}."
            )

        model = self.get_model(model_name, model_kwargs)

        # Create a new logger
        logger = configure(
            f"{logger_dir}/{model_name}",
            logger_outputs,
        )

        # Set new logger
        model.set_logger(logger)

        # Train the model
        trained_model = self.agent.train_model(
            model=model, tb_log_name=model_name, total_timesteps=total_timesteps
        )
        self.trained_model = trained_model
        return trained_model

    def save_model(self, model_name: str, directory: str, filename: str):
        """
        Saves the trained DRL model.
        :param directory: Directory where the model will be saved.
        :param filename: Name of the file to save the model.
        """
        self.trained_model.save(f"{directory}/{filename}_{model_name}")

    def load_model(
        self,
        model_name: str,
        directory: str,
        filename: str,
    ) -> A2C | PPO | DDPG | TD3 | SAC:
        """
        Loads a trained DRL model.
        :param model_name: The name of the DRL model to be loaded.
        :param directory: Directory where the model is saved.
        :param filename: Name of the file to load the model from.
        :return: The loaded DRL model.
        :raises ValueError: If the model name is not supported.
        """
        if model_name == "a2c":
            model = A2C.load(f"{directory}/{filename}_{model_name}")
        elif model_name == "ppo":
            model = PPO.load(f"{directory}/{filename}_{model_name}")
        elif model_name == "ddpg":
            model = DDPG.load(f"{directory}/{filename}_{model_name}")
        elif model_name == "td3":
            model = TD3.load(f"{directory}/{filename}_{model_name}")
        elif model_name == "sac":
            model = SAC.load(f"{directory}/{filename}_{model_name}")
        else:
            raise ValueError(f"Model {model_name} is not supported.")
        self.trained_model = model
        return model

    def test_model(
        self,
        test_env: StockTradingEnv,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Tests the trained DRL model.
        :param test_env: The environment in which the model will be tested.
        :return: A tuple containing the test results DataFrame and the actions DataFrame.
        """
        return self.agent.DRL_prediction(self.trained_model, test_env)
