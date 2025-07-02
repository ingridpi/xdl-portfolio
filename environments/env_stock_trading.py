from datetime import datetime
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs


class StockTradingEnv(gym.Env):
    def __init__(
        self,
        data: pd.DataFrame,
        stock_dimension: int,
        max_holding: int,
        initial_amount: float,
        stock_shares: List[int],
        cost_pct: float,
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicators: Optional[List[str]],
        verbose: int = 10,
        day: int = 0,
        initial: bool = True,
        prev_state: Optional[List[float]] = None,
        model_name: str = "",
        mode: str = "",
        iteration: int = 0,
        seed: Optional[int] = None,
    ):
        """
        Initialises the stock trading environment.
        :param data: DataFrame containing stock data.
        :param stock_dimension: Number of stocks in the environment.
        :param max_holding: Maximum number of shares that can be held for each stock.
        :param initial_amount: Initial cash available for trading.
        :param stock_shares: Initial shares held for each stock.
        :param cost_pct: Transaction cost percentage per trade.
        :param reward_scaling: Scaling factor for the reward.
        :param state_space: Dimension of the state space.
        :param action_space: Dimension of the action space.
        :param tech_indicators: List of technical indicators to be used in the environment.
        :param verbose: Verbosity level for logging.
        :param day: Current day in the trading data.
        :param initial: Whether the environment is being initialised for the first time.
        :param prev_state: Previous state of the environment, if any.
        :param model_name: Name of the model being used.
        :param mode: Mode of the environment (e.g., training, testing).
        :param iteration: Current iteration of the environment, if any.
        """
        self.df = data
        self.day = day
        self.data = self.df.loc[self.day, :]
        self.stock_dimension = stock_dimension
        self.max_holding = max_holding
        self.initial_amount = initial_amount
        self.stock_shares = stock_shares
        self.cost_pct = cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_space,))
        self.observation_space = spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(state_space,)
        )
        self.tech_indicators = tech_indicators if tech_indicators else []
        self.initial = initial
        self.terminal = False
        self.verbose = verbose
        self.prev_state = prev_state if prev_state is not None else []
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        # Initialise state
        self.state = self.__initialise_state()

        # Initialise reward
        self.reward = 0.0

        # Initialise counter variables
        self.cost = 0.0
        self.trades = 0
        self.episode = 0

        # Initialise the memory variables
        self.asset_memory = [self.__compute_asset_value(self.state)]
        self.rewards_memory = []
        self.action_memory = []
        self.state_memory = []
        self.date_memory = [self.__get_date()]

        # Set the random seed for reproducibility
        self._seed(seed)

    def __initialise_state(self) -> List[float]:
        """
        Initialise the state of the environment.
        :return: State representation either initial or from previous state.

        The state consists of:
        - Index 0: Balance (initial amount)
        - Indices 1 to stock_dimension: Close prices of all stocks
        - Indices stock_dimension + 1 to 2 * stock_dimension + 1: Number of shares held for each stock
        - Additional indices for technical indicators (if any)
        """
        if self.initial:
            # State:
            # - Balance: Initial amount
            # - Close prices of all stocks
            # - Number of shares held for each stock
            # - Technical indicators (if any)

            state = (
                [self.initial_amount]
                + self.data.close.values.tolist()
                + self.stock_shares
            )
        else:
            # If not initial, update from previous state
            state = (
                [self.prev_state[0]]
                + self.data.close.values.tolist()
                + self.prev_state[
                    (self.stock_dimension + 1) : (2 * self.stock_dimension + 1)
                ]
            )
        if self.tech_indicators:
            for indicator in self.tech_indicators:
                state += self.data[indicator].values.tolist()

        return state

    def __get_date(self) -> datetime:
        """
        Get the current date from the data.
        """

        return self.data.date.unique()[0]

    def __compute_asset_value(self, state: List[float]) -> float:
        """
        Compute the total asset value based on the current state.
        :param state: Current state of the environment.
        :return: Total asset value.
        """
        return state[0] + sum(
            np.array(state[1 : (self.stock_dimension + 1)])
            * np.array(
                state[
                    (self.stock_dimension + 1) : (2 * self.stock_dimension + 1)
                ]
            )
        )

    def step(
        self, actions: List[float]
    ) -> Tuple[List[float], float, bool, bool, dict]:
        """
        Execute one time step within the environment.
        :param actions: List of actions to take for each stock.
        :return: Tuple containing the next state, reward, done flag, truncated flag, and additional info.
        """

        # Check if the state is terminal
        self.terminal = self.day >= self.df.index.nunique() - 1

        if self.terminal:
            # Compute the total asset value
            end_asset_value = self.__compute_asset_value(self.state)

            # Compute the total reward
            total_reward = end_asset_value - self.asset_memory[0]
            self.reward = total_reward * self.reward_scaling
            self.rewards_memory.append(self.reward)

            # Store the asset values in a dataframe
            asset_df = self.save_asset_memory()

            # Compute the Sharpe ratio
            if asset_df["daily_return"].std() != 0:
                sharpe_ratio = (
                    (252**0.5)
                    * asset_df["daily_return"].mean()
                    / asset_df["daily_return"].std()
                )

            # Compute rewards dataframe
            rewards_df = pd.DataFrame(self.rewards_memory, columns=["reward"])
            rewards_df["date"] = self.date_memory

            # Print information if verbose
            if self.episode % self.verbose == 0:
                print("=================================")
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_asset_value:0.2f}")
                print(f"total_reward: {total_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")

                if asset_df["daily_return"].std() != 0:
                    print(f"sharpe_ratio: {sharpe_ratio:0.3f}")
                print("=================================")

        else:
            # Rescale actions to the range [-max_holding, max_holding]
            actions = (
                (np.array(actions) * self.max_holding).astype(int).tolist()
            )

            # Compute total asset value
            begin_asset_value = self.__compute_asset_value(self.state)

            # Compute the actions to be taken
            actions_array = np.array(actions)
            argsort_actions = np.argsort(actions_array)
            sell_index = argsort_actions[
                : np.where(actions_array < 0)[0].shape[0]
            ]
            buy_index = argsort_actions[::-1][
                : np.where(actions_array > 0)[0].shape[0]
            ]

            # Perform the sell actions
            for index in sell_index:
                actions[index] = self.__sell_stock(index, actions[index]) * -1

            # Perform the buy actions
            for index in buy_index:
                actions[index] = self.__buy_stock(index, actions[index])

            self.action_memory.append(actions)

            # Update the state with the new actions
            self.day += 1
            self.data = self.df.loc[self.day, :]

            self.state = self.__update_state()

            # Compute the new asset value
            end_asset_value = self.__compute_asset_value(self.state)

            # Update reward
            self.reward = end_asset_value - begin_asset_value
            self.rewards_memory.append(self.reward)
            self.reward *= self.reward_scaling

            # Update the memory
            self.asset_memory.append(end_asset_value)
            self.date_memory.append(self.__get_date())
            self.state_memory.append(self.state)

        # The fourth element in the tuple is always False for this environment
        # Corresponds to whether the environment is truncated or not
        return self.state, self.reward, self.terminal, False, {}

    def __sell_stock(self, index: int, action: int) -> float:
        """
        Perform the sell actions for a specific stock, ensuring that the stock price is positive and shares are held.
        :param index: Index of the stock to sell.
        :param action: Number of shares to sell.
        :return: Number of shares sold.
        """

        # Sell if price is positive
        if self.state[index + 1] > 0:
            # Sell if asset shares held is greater than 0
            if self.state[index + self.stock_dimension + 1] > 0:

                # Number of shares to sell: Minimum between the predicted action and the shares held
                shares_to_sell = min(
                    abs(action), self.state[index + self.stock_dimension + 1]
                )

                # Compute the revenue from the sale
                sell_revenue = (
                    self.state[index + 1] * shares_to_sell * (1 - self.cost_pct)
                )

                # Update balance
                self.state[0] += sell_revenue

                # Update the number of shares held
                self.state[index + self.stock_dimension + 1] -= shares_to_sell

                # Update the cost and trades
                self.cost += (
                    self.state[index + 1] * shares_to_sell * self.cost_pct
                )
                self.trades += 1

            # If no shares held, do not sell
            else:
                shares_to_sell = 0

        # If the stock price is not positive, do not sell
        else:
            shares_to_sell = 0

        return shares_to_sell

    def __buy_stock(self, index: int, action: int) -> float:
        """
        Perform the buy actions for a specific stock, ensuring that the stock price is positive and sufficient cash is available.
        :param index: Index of the stock to buy.
        :param action: Number of shares to buy.
        :return: Number of shares bought.
        """

        # Buy if price is positive
        if self.state[index + 1] > 0:

            # Compute the cost of buying the shares
            share_buy_cost = self.state[index + 1] * (1 + self.cost_pct)

            # Compute the maximum number of shares that can be bought
            max_shares = self.state[0] // share_buy_cost

            # Number of shares to buy: Minimum between the predicted action and the maximum shares that can be bought
            shares_to_buy = min(abs(action), max_shares)

            # Compute the total cost of buying the shares
            buy_cost = (
                self.state[index + 1] * shares_to_buy * (1 + self.cost_pct)
            )

            # Update balance
            self.state[0] -= buy_cost

            # Update the number of shares held
            self.state[index + self.stock_dimension + 1] += shares_to_buy

            # Update the cost and trades
            self.cost += self.state[index + 1] * shares_to_buy * self.cost_pct
            self.trades += 1

        # If the stock price is not positive, do not buy
        else:
            shares_to_buy = 0

        return shares_to_buy

    def __update_state(self) -> List[float]:
        """
        Update the state of the environment after taking actions.
        :return: Updated state.
        """

        state = (
            [self.state[0]]
            + self.data.close.values.tolist()
            + self.state[
                (self.stock_dimension + 1) : (2 * self.stock_dimension + 1)
            ]
        )
        if self.tech_indicators:
            for indicator in self.tech_indicators:
                state += self.data[indicator].values.tolist()

        return state

    def reset(self, *, seed=None, options=None) -> Tuple[List[float], dict]:
        """
        Resets the environment and returns a new state representation.
        :param seed: Random seed for reproducibility.
        :param options: Options for resetting the environment.
        :return: Tuple containing the initial state and an empty dictionary for additional info.
        """
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self.__initialise_state()

        self.asset_memory = [self.__compute_asset_value(self.state)]

        self.cost = 0.0
        self.trades = 0
        self.terminal = False

        self.rewards_memory = []
        self.action_memory = []
        self.state_memory = []
        self.date_memory = [self.__get_date()]

        self.episode += 1

        return self.state, {}

    def render(self, mode="human", close=False) -> List[float]:
        """
        Render the current state of the environment.
        :param mode: The rendering mode (default is "human").
        :param close: Whether to close the rendering window (not used in this environment).
        :return: Current state.
        """

        return self.state

    def get_sb_env(self) -> Tuple[DummyVecEnv, np.ndarray | VecEnvObs]:
        """
        Get the stable-baselines environment.
        :return: Stable-baselines environment and observation space.
        """

        env = DummyVecEnv([lambda: self])
        obs = env.reset()
        return env, obs

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
        Returns a DataFrame containing the actions memory.
        :return: DataFrame with actions memory.
        """

        df_actions = pd.DataFrame(
            self.action_memory,
            columns=self.data.tic.values,
        )
        df_actions["date"] = self.date_memory[:-1]

        return df_actions

    def _seed(self, seed=None):
        """
        Set the random seed for the environment.
        :param seed: Random seed for reproducibility.
        :return: List containing the seed used.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class StockTradingEnvWrapper:
    def __init__(
        self,
        train_data: pd.DataFrame,
        trade_data: pd.DataFrame,
        indicators: Optional[List[str]] = None,
        transaction_cost: float = 0.001,
        initial_cash: float = 1000000,
        max_shares: int = 100,
        reward_scaling: float = 1e-4,
    ):
        """
        Initialises the trading environment.
        :param train_data: DataFrame containing training data.
        :param trade_data: DataFrame containing trading data.
        :param indicators: List of technical indicators to be used in the environment.
        :param transaction_cost: Transaction cost per trade.
        :param initial_cash: Initial cash available for trading.
        :param max_shares: Maximum number of shares that can be held.
        :param reward_scaling: Scaling factor for the reward.
        """
        self.stock_dim = train_data.tic.nunique()
        self.state_space = 1 + 2 * self.stock_dim
        if indicators:
            self.state_space += len(indicators) * self.stock_dim
        else:
            indicators = []

        self.train_data = train_data
        self.trade_data = trade_data

        # Initial shares
        init_shares = [0] * self.stock_dim

        self.env_args = {
            "max_holding": max_shares,
            "initial_amount": initial_cash,
            "stock_shares": init_shares,
            "cost_pct": transaction_cost,
            "state_space": self.state_space,
            "stock_dimension": self.stock_dim,
            "action_space": self.stock_dim,
            "reward_scaling": reward_scaling,
            "tech_indicators": indicators,
            "initial": True,
        }

        print(
            f"Environment successfully created with \n\tStock dimension: {self.stock_dim} \n\tState space: {self.state_space}"
        )

    def get_train_env(self) -> DummyVecEnv:
        """
        Creates and returns the training environment.
        :return: Training environment instance.
        """
        self.train_env = StockTradingEnv(data=self.train_data, **self.env_args)
        return self.train_env.get_sb_env()[0]

    def get_trade_env(
        self,
    ) -> tuple[StockTradingEnv, tuple[DummyVecEnv, VecEnvObs]]:
        """
        Creates and returns the trading environment.
        :return: Trading environment instance and its stable-baselines environment.
        """
        self.trade_env = StockTradingEnv(data=self.trade_data, **self.env_args)
        return self.trade_env, self.trade_env.get_sb_env()
