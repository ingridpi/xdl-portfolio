from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
import pandas as pd


class FinRLTradingEnv:
    def __init__(
        self,
        train_data: pd.DataFrame,
        trade_data: pd.DataFrame,
        indicators: list,
        transaction_cost: float = 0.001,
        initial_cash: float = 1000000,
        max_shares: int = 100,
        reward_scaling: float = 1e-4,
    ):
        """
        Initialises the FinRL trading environment.
        :param train_data: DataFrame containing training data.
        :param trade_data: DataFrame containing trading data.
        :param indicators: List of technical indicators to be used in the environment.
        :param transaction_cost: Transaction cost per trade.
        :param initial_cash: Initial cash available for trading.
        :param max_shares: Maximum number of shares that can be held.
        :param reward_scaling: Scaling factor for the reward.
        """
        self.stock_dim = train_data.tic.nunique()
        self.state_space = (
            1 + 2 * self.stock_dim + len(indicators) * self.stock_dim
        )

        self.train_data = train_data
        self.trade_data = trade_data

        # Transaction cost
        cost_list = [transaction_cost] * self.stock_dim

        # Initial shares
        init_shares = [0] * self.stock_dim

        self.env_args = {
            "hmax": max_shares,
            "initial_amount": initial_cash,
            "num_stock_shares": init_shares,
            "buy_cost_pct": cost_list,
            "sell_cost_pct": cost_list,
            "state_space": self.state_space,
            "stock_dim": self.stock_dim,
            "action_space": self.stock_dim,
            "reward_scaling": reward_scaling,
            "tech_indicator_list": indicators,
        }

        print(
            f"Environment successfully created with \n\tStock dimension: {self.stock_dim} \n\tState space: {self.state_space}"
        )

    def get_train_env(self) -> DummyVecEnv:
        """
        Creates and returns the training environment.
        :return: Training environment instance.
        """
        self.train_env = StockTradingEnv(df=self.train_data, **self.env_args)
        return self.train_env.get_sb_env()[0]

    def get_trade_env(
        self,
    ) -> tuple[StockTradingEnv, tuple[DummyVecEnv, VecEnvObs]]:
        """
        Creates and returns the trading environment.
        :return: Trading environment instance and its stable-baselines environment.
        """
        self.trade_env = StockTradingEnv(df=self.trade_data, **self.env_args)
        return self.trade_env, self.trade_env.get_sb_env()
