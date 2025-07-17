from typing import Dict, Literal

import pandas as pd
from finrl.plot import convert_daily_return_to_pyfolio_ts
from pyfolio import timeseries
from pypfopt import EfficientFrontier, expected_returns, risk_models

from config import config


class PortfolioBenchmark:
    def __init__(self) -> None:
        """
        Initialise the PortfolioBenchmark class.
        """
        pass

    def set_data(
        self,
        train_data: pd.DataFrame,
        trade_data: pd.DataFrame,
        lookback: int = 252,
    ) -> None:
        """
        Combine the training and testing data into a single DataFrame and
        filter the data for the specified time period.
        :param train_data: DataFrame containing training data.
        :param trade_data: DataFrame containing trading data.
        :param lookback: Lookback period in days for the test start date.
        """
        data = pd.concat([train_data, trade_data], ignore_index=True)

        # Find the test start date or the next available trading date
        test_start = pd.to_datetime(config.TEST_START_DATE)
        if test_start not in data["date"].values:
            print("Test start date is not a trading date in the dataset.")
            next_date = test_start
            max_lookahead = 5  # Maximum number of days to look ahead
            lookahead = 1
            while lookahead <= max_lookahead:
                next_date = test_start + pd.Timedelta(days=lookahead)
                if next_date in data["date"].values:
                    print(
                        f"Using next available trading date: {next_date.date()}"
                    )
                    test_start = next_date
                    break
                lookahead += 1
            else:
                raise ValueError(
                    "No available trading date found after the test start date."
                )

        data.set_index(data["date"].astype("category").cat.codes, inplace=True)

        # Filter data for the test period plus lookback window
        start_idx = (
            data[data["date"] == test_start].index.unique().values[0] - lookback
        )
        end_idx = data.index.max()

        self.data = data.loc[start_idx:end_idx, :]
        self.lookback = lookback

    def optimise_portfolio(
        self,
        strategy: Literal["mean", "min", "momentum", "equal"],
        initial_capital: int = 100000,
    ) -> pd.DataFrame:
        """
        Optimises the portfolio depending on the specified strategy:
        - 'mean': Mean-variance optimisation
        - 'min': Minimum variance optimisation
        - 'momentum': Momentum-based strategy
        - 'equal': Equal weight strategy
        :param strategy: Type of optimisation
        :param initial_capital: Initial capital for the portfolio, default is 100000.
        :return: DataFrame with date, account_value, and daily_return columns.
        """
        # Pivot data: date as index, tics as columns
        price_df = self.data.pivot(
            index="date", columns="tic", values="close"
        ).sort_index()

        # Calculate daily returns
        returns_df = price_df.pct_change().dropna()

        # Initialise memory
        weights_record = {}
        portfolio_returns = [0.0]
        portfolio_values = [initial_capital]
        portfolio_value = initial_capital

        dates = returns_df.index

        for i in range(self.lookback + 1, len(returns_df)):
            current_date = dates[i]

            window_prices = price_df.iloc[i - self.lookback : i + 1]

            if strategy == "momentum":
                weights = self.__momentum_strategy(window_prices)
            elif strategy == "equal":
                weights = self.__equal_weight_strategy(
                    price_df.columns.tolist()
                )
            elif strategy == "mean":
                weights = self.__mean_variance_strategy(window_prices)
            elif strategy == "min":
                weights = self.__min_variance_strategy(window_prices)
            else:
                raise ValueError(
                    "Invalid strategy. Choose from 'mean', 'min', 'momentum', or 'equal'."
                )

            weights_record[current_date] = weights

            # Calculate daily return of the portfolio
            daily_return = sum(
                weights.get(tic, 0) * returns_df.iloc[i][tic]
                for tic in price_df.columns
            )
            portfolio_returns.append(daily_return)

            # Calculate daily portfolio value
            portfolio_value *= 1 + daily_return
            portfolio_values.append(portfolio_value)

        # Convert to pandas DataFrame with date column, account_value and daily_return columns
        portfolio = pd.DataFrame(
            {
                "date": dates[self.lookback :],
                "account_value": portfolio_values,
                "daily_return": portfolio_returns,
            }
        )

        return portfolio

    def __momentum_strategy(
        self, prices: pd.DataFrame, lookback: int = 252
    ) -> Dict:
        """
        Implements a simple momentum strategy.
        :param prices: DataFrame of prices.
        :return: Dictionary of weights based on momentum.
        """

        returns = prices.pct_change(lookback).iloc[-1]

        # Clipping negative returns to zero
        returns = returns.clip(lower=0)

        # If all returns are zero, return equal weights
        if returns.sum() == 0:
            return {tic: 1 / len(prices.columns) for tic in prices.columns}
        # Normalise weights to sum to 1
        weights = returns / returns.sum()
        return weights.to_dict()

    def __equal_weight_strategy(self, ticker_list: list) -> Dict:
        """
        Implements an equal weight strategy.
        :param ticker_list: List of asset tickers.
        :return: Dictionary of equal weights for each asset.
        """
        num_assets = len(ticker_list)
        return {tic: 1 / num_assets for tic in ticker_list}

    def __mean_variance_strategy(self, prices: pd.DataFrame) -> Dict:
        """
        Implements a mean-variance optimisation strategy.
        :param prices: DataFrame of prices.
        :return: Dictionary of weights based on mean-variance optimisation.
        """
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)

        ef = EfficientFrontier(mu, S)
        _ = ef.max_sharpe()
        return ef.clean_weights()

    def __min_variance_strategy(self, prices: pd.DataFrame) -> Dict:
        """
        Implements a minimum variance optimisation strategy.
        :param prices: DataFrame of prices.
        :return: Dictionary of weights based on minimum variance optimisation.
        """
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)

        ef = EfficientFrontier(mu, S)
        _ = ef.min_volatility()
        return ef.clean_weights()

    def compute_perf_stats(self, df_account: pd.DataFrame) -> pd.Series:
        pf_returns = convert_daily_return_to_pyfolio_ts(df_account)
        perf_stats_alg = timeseries.perf_stats(
            returns=pf_returns,
            factor_returns=pf_returns,
            positions=None,
            transactions=None,
            turnover_denom="AGB",
        )
        return perf_stats_alg
