from datetime import datetime
import itertools
import pandas as pd
from typing import List
import exchange_calendars as ecals


class FinancialDataPreprocessor:
    def __init__(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date

    def preprocess(self, data: pd.DataFrame, exchange: str) -> pd.DataFrame:
        """
        Preprocess financial data by cleaning it and adding additional features.
        """
        df = data.copy()

        df = self.__clean_data(df, exchange)

        # Add day of the week column
        df["DayOfWeek"] = df["Date"].dt.dayofweek

        return df.reset_index(drop=True)

    def __clean_data(self, data: pd.DataFrame, exchange: str) -> pd.DataFrame:
        """
        Clean the financial data by ensuring all trading days are represented
        and filling missing values appropriately.
        """
        trading_days = self.__get_trading_dates(exchange)
        tickers = data["Ticker"].unique()

        # Create a DataFrame with all combinations of trading dates and tickers
        index = list(itertools.product(trading_days, tickers))
        df = pd.DataFrame(index, columns=["Date", "Ticker"]).merge(
            data, on=["Date", "Ticker"], how="left"
        )

        df = df.sort_values(by=["Ticker", "Date"])

        # Fill missing values with forward fill method and then backward fill
        df["Open"] = df.groupby("Ticker")["Open"].ffill().bfill()
        df["Close"] = df.groupby("Ticker")["Close"].ffill().bfill()
        df["High"] = df.groupby("Ticker")["High"].ffill().bfill()
        df["Low"] = df.groupby("Ticker")["Low"].ffill().bfill()

        # Fill missing volumes with 0 to indicate no trading activity
        df["Volume"] = df["Volume"].fillna(0)

        return df

    def __get_trading_dates(self, exchange: str) -> List[datetime]:
        """
        Get trading dates for a given exchange within a specified date range.
        """

        dates = (
            ecals.get_calendar(exchange)
            .sessions_in_range(start=self.start_date, end=self.end_date)
            .to_list()
        )

        # Convert to datetime objects
        return [d.to_pydatetime() for d in dates]

