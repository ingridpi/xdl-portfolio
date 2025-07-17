from typing import Literal, Optional, Tuple

import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.tb_callback import TensorboardCallback
from config import config_models


class DRLAgent:
    def __init__(self):
        """
        Initialises the DRL agent.
        """

    def get_model(
        self,
        model_name: str,
        environment: DummyVecEnv,
        directory: str,
        use_case: Literal["stock-trading", "portfolio-optimisation"],
        model_kwargs: Optional[dict] = None,
        policy: str = "MlpPolicy",
        policy_kwargs: Optional[dict] = None,
        seed: Optional[int] = None,
        verbose: int = 0,
    ) -> BaseAlgorithm:
        """
        Returns a DRL model based on the specified model name and parameters.
        :param model_name: The name of the DRL model to create.
        :param environment: The environment in which the model will be trained.
        :param directory: The directory where the tensorboard logs will be saved.
        :param model_kwargs: Additional keyword arguments for the model.
        :param policy: The policy to use for the model.
        :param policy_kwargs: Additional keyword arguments for the policy.
        :param seed: Random seed for reproducibility.
        :param verbose: Verbosity level for the model.
        :return: An instance of the specified DRL model.
        :raises ValueError: If the model name is not supported.
        """

        if model_name not in config_models.MODELS:
            raise ValueError(
                f"Model {model_name} is not supported. Choose from {config_models.MODELS.keys()}."
            )

        if model_kwargs is None:
            if use_case == "stock-trading":
                model_kwargs = config_models.MODEL_KWARGS_STOCK[model_name]
            elif use_case == "portfolio-optimisation":
                model_kwargs = config_models.MODEL_KWARGS_PORTFOLIO[model_name]

        print(f"Model arguments: {model_kwargs}")

        tensorboard_log = f"{directory}/{model_name}"

        logger = configure(tensorboard_log, ["csv", "tensorboard"])

        model = config_models.MODELS[model_name](
            policy=policy,
            env=environment,
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
        model: BaseAlgorithm,
        tb_log_name: str,
        log_interval: int = 20,
        total_timesteps: int = 100000,
    ) -> BaseAlgorithm:
        """
        Trains the DRL model with the specified parameters.
        :param model: The DRL model to train.
        :param tb_log_name: The name for the tensorboard log.
        :param log_interval: The interval for logging training progress.
        :param total_timesteps: The total number of timesteps for training.
        :return: The trained DRL model.
        """

        model = model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            callback=TensorboardCallback(),
        )

        return model

    def predict(
        self,
        model: BaseAlgorithm,
        environment,
        deterministic: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Makes a prediction given the provided model and environment.
        :param model: The trained DRL model.
        :param environment: The environment in which to run the prediction.
        :param deterministic: Whether to use deterministic actions.
        :return: The account memory and actions memory after the prediction.
        """
        test_env, test_obs = environment.get_sb_env()
        account_memory = []
        actions_memory = []

        test_env.reset()
        max_steps = environment.df.index.nunique()

        for i in range(max_steps):
            action, _ = model.predict(test_obs, deterministic=deterministic)

            # Last step: Save account and actions memory
            if i == max_steps - 1:
                account_memory = test_env.env_method(
                    method_name="save_asset_memory"
                )
                actions_memory = test_env.env_method(
                    method_name="save_action_memory"
                )

            test_obs, _, dones, _ = test_env.step(action)

            # If the environment is done, break the loop
            if dones[0]:
                break

        return account_memory[0], actions_memory[0]

    def save_model(
        self,
        model: BaseAlgorithm,
        model_name: str,
        directory: str,
        filename: str,
    ):
        """
        Saves the trained model to a specified directory.
        :param model: The trained DRL model.
        :param model_name: The name of the model.
        :param directory: The directory where the model will be saved.
        :param filename: The filename for the saved model.
        """
        model_path = f"{directory}/{filename}_{model_name}"
        model.save(model_path)
        print(f"Model saved to {model_path}")

    def load_model(
        self,
        model_name: str,
        directory: str,
        filename: str,
    ) -> BaseAlgorithm:
        """
        Loads a trained model from a specified directory.
        :param model_name: The name of the model to load.
        :param directory: The directory where the model is saved.
        :param filename: The filename of the saved model.
        :return: The loaded DRL model.
        :raises ValueError: If the model name is not supported.
        """
        model_path = f"{directory}/{filename}_{model_name}"

        if model_name not in config_models.MODELS:
            raise ValueError(
                f"Model {model_name} is not supported. Choose from {config_models.MODELS.keys()}."
            )

        model = config_models.MODELS[model_name].load(model_path)

        print(f"Model successfully loaded from {model_path}")

        return model
