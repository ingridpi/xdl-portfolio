from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

from config import config


class PortfolioOptimisationEnv(gym.Env):
    def __init__(
        self,
        data: pd.DataFrame,
        stock_dimension: int,
        initial_amount: float,
        reward_scaling: float,
        state_space: int,
        action_space: int,
        state_columns: List[str],
        normalisation_strategy: Literal["sum", "softmax"],
        verbose: int = 10,
        day: int = 0,
        seed: Optional[int] = None,
    ):
        """
        Initialise the portfolio optimisation environment.
        :param data: DataFrame containing stock data.
        :param stock_dimension: Number of stocks in the environment.
        :param initial_amount: Initial cash available for portfolio allocation.
        :param reward_scaling: Scaling factor for the reward.
        :param state_space: Dimension of the state space.
        :param action_space: Dimension of the action space.
        :param state_columns: List of columns in the dataframe to represent the state space.
        :param normalisation_strategy: Strategy (softmax, sum) to normalise the actions to sum to 1.
        :param verbose: Verbosity level for logging.
        :param day: Current day in the trading data.
        :param seed: Random seed for reproducibility.
        """
        self.df = data
        self.day = day
        self.data = self.df.loc[self.day, :]
        self.stock_dimension = stock_dimension
        self.initial_amount = initial_amount
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = spaces.Box(low=0, high=1, shape=(action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.state_space,
                self.stock_dimension,
            ),
        )
        self.state_columns = state_columns
        self.terminal = False
        self.verbose = verbose
        self.normalisation_strategy = normalisation_strategy

        # Initialise state
        self.state = self.__get_state()
        self.portfolio_value = self.initial_amount
        self.weights = [1 / self.stock_dimension] * self.stock_dimension

        # Initialise reward
        self.reward = 0.0

        # Initialise counter variables
        self.episode = 0

        # Initialise memory variables
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = [self.reward]
        self.return_memory = [0]
        self.actions_memory = [self.weights]
        self.date_memory = [self.__get_date()]

        # Set the random seed for reproducibility
        self._seed(seed)

    def __get_state(self) -> List:
        """
        Get the current state representation from the data.
        :return: Current state as a list of values for each column in state_columns.
        """
        return [self.data[col].values.tolist() for col in self.state_columns]

    def __get_date(self) -> datetime:
        """
        Get the current date from the data.
        :return: Current date.
        """

        return self.data.date.unique()[0]

    def step(self, action: np.ndarray) -> Tuple[List, float, bool, bool, Dict]:
        """
        Execute one time step within the environment.
        :param action: List of actions to take for each stock.
        :return: Tuple containing the next state, reward, done flag, truncated flag, and additional info.
        """
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            df_return = pd.DataFrame(self.return_memory)
            df_return.columns = ["daily_return"]

            # Print information if verbose
            if self.episode % self.verbose == 0:
                print("=================================")
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset:{self.asset_memory[0]:.2f}")
                print(f"end_total_asset:{self.portfolio_value:.2f}")
                if df_return["daily_return"].std() != 0:
                    sharpe = (
                        (252**0.5)
                        * df_return["daily_return"].mean()
                        / df_return["daily_return"].std()
                    )
                    print(f"sharpe_ratio: {sharpe:.2f}")
                print("=================================")

        else:
            # Normalise actions
            weights = self.__normalise_actions(
                action, self.normalisation_strategy  # type: ignore
            )
            self.actions_memory.append(weights.tolist())

            # Retrieve previous day's prices
            prev_prices = np.array(self.data.close.values)

            # Update the state with the new actions
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.state = self.__get_state()

            # Retrieve current day's prices
            curr_prices = np.array(self.data.close.values)

            # Compute the portfolio returns
            portfolio_return = np.dot(
                ((curr_prices / prev_prices) - 1), weights
            )

            # Update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)

            # Update the reward: Change in portfolio value
            self.reward = new_portfolio_value - self.portfolio_value
            self.portfolio_value = new_portfolio_value
            self.rewards_memory.append(self.reward)
            self.reward *= self.reward_scaling

            # Update the memory
            self.return_memory.append(portfolio_return)
            self.date_memory.append(self.__get_date())
            self.asset_memory.append(new_portfolio_value)

        # The fourth element in the tuple is always False for this environment
        # Corresponds to whether the environment is truncated or not
        return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ) -> Tuple[List, Dict]:
        """
        Resets the environment and returns a new state representation.
        :param seed: Random seed for reproducibility.
        :param options: Options for resetting the environment.
        :return: Tuple containing the initial state and an empty dictionary for additional info.
        """
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self.__get_state()
        self.portfolio_value = self.initial_amount

        self.terminal = False

        self.weights = [1 / self.stock_dimension] * self.stock_dimension
        self.asset_memory = [self.initial_amount]
        self.return_memory = [0]
        self.rewards_memory = [0.0]
        self.actions_memory = [self.weights]
        self.date_memory = [self.__get_date()]

        self.episode += 1

        return self.state, {}

    def render(self, mode: str = "human", close=False) -> List:
        """
        Render the current state of the environment.
        :param mode: The rendering mode (default is "human").
        :param close: Whether to close the rendering window (not used in this environment).
        :return: Current state.
        """
        return self.state

    def __normalise_actions(
        self,
        actions: np.ndarray,
        strategy: Literal["sum", "softmax"],
    ) -> np.ndarray:
        """
        Apply normalisation to the actions.
        :param actions: Actions to be normalised.
        :param strategy: Normalisation strategy to use.
        :return: Normalised actions.
        """
        if strategy == "sum":
            total_sum = np.sum(actions)
            if total_sum != 0:
                return actions / total_sum
            else:
                # Explicitly handle zero-sum case by returning uniform distribution
                return np.ones_like(actions) / len(actions)
        else:
            exp_actions = np.exp(actions)
            return exp_actions / np.sum(exp_actions)

    def save_asset_memory(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing the asset memory.
        :return: DataFrame with asset memory.
        """
        df_asset = pd.DataFrame(
            {"date": self.date_memory, "account_value": self.asset_memory}
        )
        df_asset["daily_return"] = (
            df_asset["account_value"].pct_change().fillna(0)
        )

        return df_asset

    def save_action_memory(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing the action memory.
        :return: DataFrame with actions memory.
        """
        df_actions = pd.DataFrame(
            self.actions_memory,
            columns=self.data.tic.values,
        )
        df_actions["date"] = self.date_memory

        return df_actions

    def _seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Set the random seed for the environment.
        :param seed: Random seed for reproducibility.
        :return: List containing the seed used.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self) -> Tuple[DummyVecEnv, VecEnvObs]:
        """
        Get the stable-baselines environment.
        :return: Stable-baselines environment and observation space.
        """
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs


class PortfolioOptimisationEnvWrapper:
    def __init__(
        self,
        train_data: pd.DataFrame,
        trade_data: pd.DataFrame,
        state_columns: List[str] = ["close"],
        initial_amount: float = config.INITIAL_AMOUNT,
        reward_scaling: float = 1e-1,
        normalisation_strategy: Literal["sum", "softmax"] = "softmax",
    ):
        """
        Initialises the trading environment.
        :param train_data: DataFrame containing training data.
        :param trade_data: DataFrame containing trading data.
        :param state_columns: List of columns to represent the state space.
        :param initial_amount: Initial cash available for trading.
        :param reward_scaling: Scaling factor for the reward.
        :param normalisation_strategy: Strategy (softmax, sum) to normalise the actions to sum to 1.
        """
        self.stock_dim = train_data.tic.nunique()
        self.state_space = len(state_columns)
        self.train_data = train_data
        self.trade_data = trade_data

        self.env_args = {
            "initial_amount": initial_amount,
            "state_space": self.state_space,
            "stock_dimension": self.stock_dim,
            "action_space": self.stock_dim,
            "reward_scaling": reward_scaling,
            "state_columns": state_columns,
            "normalisation_strategy": normalisation_strategy,
        }

        print(
            f"Environment successfully created with \n\tStock dimension: {self.stock_dim} \n\tState space: {self.state_space}"
        )

    def get_train_env(self) -> DummyVecEnv:
        """
        Creates and returns the training environment.
        :return: Training environment instance.
        """
        self.train_env = PortfolioOptimisationEnv(
            data=self.train_data, **self.env_args
        )
        return self.train_env.get_sb_env()[0]

    def get_trade_env(
        self,
    ) -> Tuple[PortfolioOptimisationEnv, Tuple[DummyVecEnv, VecEnvObs]]:
        """
        Creates and returns the trading environment.
        :return: Trading environment instance and its stable-baselines environment.
        """
        self.trade_env = PortfolioOptimisationEnv(
            data=self.trade_data, **self.env_args
        )
        return self.trade_env, self.trade_env.get_sb_env()
