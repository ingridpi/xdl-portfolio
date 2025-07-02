from typing import Optional, Tuple

import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.tb_callback import TensorboardCallback
from config import config, config_models
from environments.env_stock_trading import StockTradingEnv


class DRLAgent:
    def __init__(self, env: DummyVecEnv):
        """
        Initialises the DRL agent with the given environment.
        :param env: The environment for the agent.
        """
        self.env = env

    def get_model(
        self,
        model_name: str,
        directory: str,
        model_kwargs: Optional[dict] = None,
        policy: str = "MlpPolicy",
        policy_kwargs: Optional[dict] = None,
        seed: Optional[int] = None,
        verbose: int = 0,
    ) -> BaseAlgorithm:
        """
        Returns a DRL model based on the specified model name and parameters.
        :param model_name: The name of the DRL model to be used.
        :param model_kwargs: Additional keyword arguments for the model.
        :param policy: The policy to be used by the model.
        :param policy_kwargs: Additional keyword arguments for the policy.
        :return: A DRL model instance.
        """
        if model_name not in config_models.MODELS:
            raise ValueError(
                f"Model {model_name} is not supported. Choose from {config.MODELS}."
            )

        if model_kwargs is None:
            model_kwargs = config_models.MODEL_KWARGS[model_name]

        tensorboard_log = f"{directory}/{model_name}"

        logger = configure(tensorboard_log, ["csv", "tensorboard"])

        model = config_models.MODELS[model_name](
            policy=policy,
            env=self.env,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            seed=seed,
            policy_kwargs=policy_kwargs,
            **model_kwargs,
        )

        model.set_logger(logger)

        return model

    def train(
        self,
        model_name: str,
        directory: str,
        tb_log_name: str,
        log_interval: int = 20,
        total_timesteps: int = 100000,
    ) -> BaseAlgorithm:

        model = self.get_model(model_name=model_name, directory=directory)

        model = model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            callback=TensorboardCallback(),
        )

        self.model = model

        return model

    def predict(
        self,
        model: BaseAlgorithm,
        test_env: StockTradingEnv,
        deterministic: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Obtain state observation
        test_env_gym, test_obs = test_env.get_sb_env()

        # Outputs
        account_memory = []
        actions_memory = []

        test_env_gym.reset()

        # Iterate over testing data
        for i in range(test_env.df.index.nunique()):
            # Predict action to take
            action, _ = model.predict(test_obs, deterministic=deterministic)  # type: ignore

            # Save the predictions
            if i == test_env.df.index.nunique() - 1:
                account_memory = test_env_gym.env_method(
                    method_name="save_asset_memory"
                )
                actions_memory = test_env_gym.env_method(
                    method_name="save_action_memory"
                )

            # Perform the predicted action
            test_obs, _, done, _ = test_env_gym.step(action)

            if done[0]:
                break

        return account_memory[0], actions_memory[0]

    def save_model(
        self,
        model_name: str,
        directory: str,
        filename: str,
    ):
        """
        Saves the trained DRL model.
        :param model: The DRL model to be saved.
        :param directory: Directory where the model will be saved.
        :param filename: Name of the file to save the model.
        """
        self.model.save(f"{directory}/{filename}_{model_name}")

    def load_model(
        self,
        model_name: str,
        directory: str,
        filename: str,
    ) -> BaseAlgorithm:
        """
        Loads a trained DRL model.
        :param model_name: The name of the DRL model to be loaded.
        :param directory: Directory where the model is saved.
        :param filename: Name of the file to load the model from.
        :return: The loaded DRL model.
        :raises ValueError: If the model name is not supported.
        """
        if model_name not in config_models.MODELS:
            raise ValueError(f"Model {model_name} is not supported.")

        model = config_models.MODELS[model_name].load(
            f"{directory}/{filename}_{model_name}"
        )
        self.model = model
        return model
