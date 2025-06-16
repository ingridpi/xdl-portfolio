from datetime import datetime
import itertools
import pandas as pd
from typing import List
import exchange_calendars as ecals
from stockstats import StockDataFrame

from preprocessor.findata_downloader import FinancialDataDownloader


class FinancialDataPreprocessor:
    def __init__(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date

    def preprocess(
        self,
        data: pd.DataFrame,
        exchange: str,
        use_tech_indicators: bool = False,
        tech_indicators: List[str] = [],
        use_macro_indicators: bool = False,
        macro_indicators: List[str] = [],
    ) -> pd.DataFrame:
        """
        Preprocess financial data by cleaning it and adding additional features.
        """
        df = data.copy()

        df = self.__clean_data(df, exchange)

        # Add day of the week column
        df["DayOfWeek"] = df["Date"].dt.dayofweek

        if use_tech_indicators:
            df = self.__add_technical_indicators(df, tech_indicators)

        if use_macro_indicators:
            df = self.__add_macroeconomic_indicators(df, macro_indicators)

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

    def split_train_test(
        self, data: pd.DataFrame, train_end_date: pd.Timestamp | str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and testing sets based on the given train end date.
        """

        # Ensure train end date is part of the DataFrame
        train_end_date = pd.to_datetime(train_end_date)
        if "Date" not in data.columns:
            raise ValueError("Data must contain a 'Date' column for splitting.")
        if (
            train_end_date < data["Date"].min()
            or train_end_date > data["Date"].max()
        ):
            raise ValueError(
                f"The train end date is outside of the 'Date' column values."
            )

        # Split the data into training and testing sets
        train_data = data[data["Date"] <= train_end_date].copy()
        test_data = data[data["Date"] > train_end_date].copy()

        return train_data, test_data

    def __add_technical_indicators(
        self, data: pd.DataFrame, indicators: List[str]
    ) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame based on the specified indicators.
        """

        # Convert the dataframe to StockDataFrame
        stock_df = StockDataFrame.retype(data.copy())
        tickers = data["Ticker"].unique()

        # Iterate over the indicators
        for indicator in indicators:
            indicator_df = pd.DataFrame()

            # Iterate over each ticker
            for ticker in tickers:

                # Extract the indicator data for the ticker
                try:
                    ind_df = stock_df[stock_df["ticker"] == ticker][indicator]
                    ind_df = pd.DataFrame(ind_df)
                    ind_df["Ticker"] = ticker
                    ind_df["Date"] = data[data["Ticker"] == ticker][
                        "Date"
                    ].values

                    indicator_df = pd.concat(
                        [indicator_df, ind_df], ignore_index=True
                    )
                except Exception as e:
                    print(
                        f"Error processing indicator '{indicator}' for ticker '{ticker}': {e}"
                    )

            data = data.merge(indicator_df, on=["Ticker", "Date"], how="left")

        data.fillna(0, inplace=True)  # Fill NaN values with 0

        return data

    def __add_macroeconomic_indicators(
        self, data: pd.DataFrame, indicators: List[str]
    ) -> pd.DataFrame:
        """
        Add macroeconomic indicators to the DataFrame based on the specified indicators.
        """
        # Placeholder for macroeconomic indicators
        # In a real implementation, this would fetch and merge actual macroeconomic data

        findownloader = FinancialDataDownloader(self.start_date, self.end_date)

        for indicator in indicators:
            if indicator == "^VIX":
                vix = findownloader.download_data([indicator])
                indicator_df = vix[["Date", "Close"]].rename(
                    columns={"Close": indicator}
                )

                data = data.merge(indicator_df, on=["Date"])

        return data
