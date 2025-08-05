import itertools
from datetime import datetime
from typing import List, Optional

import exchange_calendars as ecals
import pandas as pd
from stockstats import StockDataFrame

from preprocessor.findata_downloader import FinancialDataDownloader


class FinancialDataPreprocessor:
    def __init__(self, start_date: str, end_date: str) -> None:
        """
        Initialses the FinancialDataPreprocessor with a date range.
        :param start_date: Start date for the data processing in 'YYYY-MM-DD' format.
        :param end_date: End date for the data processing in 'YYYY-MM-DD' format.
        """
        self.start_date = start_date
        self.end_date = end_date

    def preprocess(
        self,
        data: pd.DataFrame,
        exchange: str,
        use_tech_indicators: bool = False,
        tech_indicators: Optional[List[str]] = None,
        use_macro_indicators: bool = False,
        macro_indicators: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Preprocess financial data by cleaning it and adding additional features.
        :param data: DataFrame containing financial data with columns ['date', 'tic', 'open', 'high', 'low', 'close', 'volume'].
        :param exchange: The stock exchange for which the data is being processed (e.g., 'NYSE', 'LSE').
        :param use_tech_indicators: Whether to add technical indicators (Default is False).
        :param tech_indicators: List of technical indicators to add.
        :param use_macro_indicators: Whether to add macroeconomic indicators (Default is False).
        :param macro_indicators: List of macroeconomic indicators to add.
        :return: Processed DataFrame with additional features.
        """
        df = data.copy()

        df = self.__clean_data(df, exchange)

        # Add day of the week column
        df["day"] = df["date"].dt.dayofweek

        if use_tech_indicators and tech_indicators:
            df = self.__add_technical_indicators(df, tech_indicators)

        if use_macro_indicators and macro_indicators:
            df = self.__add_macroeconomic_indicators(df, macro_indicators)

        df = self.__rename_columns(df)

        return df

    def __clean_data(self, data: pd.DataFrame, exchange: str) -> pd.DataFrame:
        """
        Clean the financial data by ensuring all trading days are represented
        and filling missing values appropriately.
        :param data: DataFrame containing financial data with columns ['date', 'tic', 'open', 'high', 'low', 'close', 'volume'].
        :param exchange: The stock exchange for which the data is being processed (e.g., 'NYSE', 'LSE').
        :return: Cleaned DataFrame with all trading days represented and missing values filled.
        """
        trading_days = self.__get_trading_dates(exchange)
        tickers = data["tic"].unique()

        # Create a DataFrame with all combinations of trading dates and tickers
        index = list(itertools.product(trading_days, tickers))
        df = pd.DataFrame(index, columns=["date", "tic"]).merge(
            data, on=["date", "tic"], how="left"
        )

        df = df.sort_values(by=["tic", "date"])

        # Fill missing values with forward fill method and then backward fill
        df["open"] = df.groupby("tic")["open"].ffill().bfill()
        df["close"] = df.groupby("tic")["close"].ffill().bfill()
        df["high"] = df.groupby("tic")["high"].ffill().bfill()
        df["low"] = df.groupby("tic")["low"].ffill().bfill()

        # Fill missing volumes with 0 to indicate no trading activity
        df["volume"] = df["volume"].fillna(0)

        return df

    def __get_trading_dates(self, exchange: str) -> List[datetime]:
        """
        Get trading dates for a given exchange within a specified date range.
        :param exchange: The stock exchange for which the trading dates are needed (e.g., 'NYSE', 'LSE').
        :return: List of trading dates as datetime objects.
        """
        dates = (
            ecals.get_calendar(exchange)
            .sessions_in_range(start=self.start_date, end=self.end_date)
            .to_list()
        )

        # Convert to datetime objects
        return [d.to_pydatetime() for d in dates]

    def __add_technical_indicators(
        self, data: pd.DataFrame, indicators: List[str]
    ) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame based on the specified indicators.
        :param data: DataFrame containing financial data with columns ['date', 'tic', 'open', 'high', 'low', 'close', 'volume'].
        :param indicators: List of technical indicators to add (e.g., ['macd', 'rsi', 'cci']).
        :return: DataFrame with additional technical indicators.
        :raises Exception: If there is an error processing a specific indicator for a ticker.
        """
        # Convert the dataframe to StockDataFrame
        stock_df = StockDataFrame.retype(data.copy())
        tickers = data["tic"].unique()

        # Iterate over the indicators
        for indicator in indicators:
            indicator_df = pd.DataFrame()

            # Iterate over each ticker
            for ticker in tickers:

                # Extract the indicator data for the ticker
                try:
                    ind_df = stock_df[stock_df["tic"] == ticker][indicator]
                    ind_df = pd.DataFrame(ind_df)
                    ind_df["tic"] = ticker
                    ind_df["date"] = data[data["tic"] == ticker]["date"].values

                    indicator_df = pd.concat(
                        [indicator_df, ind_df], ignore_index=True
                    )
                except Exception as e:
                    print(
                        f"Error processing indicator '{indicator}' for ticker '{ticker}': {e}"
                    )

            data = data.merge(indicator_df, on=["tic", "date"], how="left")

        data.fillna(0, inplace=True)  # Fill NaN values with 0

        return data

    def __add_macroeconomic_indicators(
        self, data: pd.DataFrame, indicators: List[str]
    ) -> pd.DataFrame:
        """
        Add macroeconomic indicators to the DataFrame based on the specified indicators.
        :param data: DataFrame containing financial data with columns ['date', 'tic', 'open', 'high', 'low', 'close', 'volume'].
        :param indicators: List of macroeconomic indicators to add (e.g., ['^VIX']).
        :return: DataFrame with additional macroeconomic indicators.
        """

        findownloader = FinancialDataDownloader(self.start_date, self.end_date)

        for indicator in indicators:
            # Add volatility index (VIX) data by downloading it from the financial data downloader
            if indicator == "^VIX":
                vix = findownloader.download_data([indicator])
                indicator_df = vix[["date", "close"]].rename(
                    columns={"close": indicator}
                )

                data = data.merge(indicator_df, on=["date"])

        return data

    def __rename_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns to a consistent format to work with FinRL library.
        :param data: DataFrame containing financial data with columns ['date', 'tic', 'open', 'high', 'low', 'close', 'volume'].
        :return: DataFrame with renamed columns.
        """

        # Convert column names to lowercase and remove non alphanumeric characters
        data.columns = data.columns.str.lower().str.replace(
            r"[^a-z0-9_]", "", regex=True
        )

        return data

    def __set_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Set the index of the DataFrame to a number from 0 to the number of distinct dates in the dataset.
        This is useful for ensuring that the index is consistent with the FinRL library's expectations.
        :param data: DataFrame containing financial data.
        :return: DataFrame with the index set to a number from 0 to the number of distinct dates.
        """

        data.sort_values(by=["date", "tic"], inplace=True)

        # Set the index to a number from 0 to number of distinct dates in the dataset
        data.set_index(data["date"].astype("category").cat.codes, inplace=True)

        return data

    def split_train_test(
        self, data: pd.DataFrame, test_start_date: pd.Timestamp | str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and testing sets based on the given test start date.
        :param data: DataFrame containing financial data.
        :param test_start_date: The start date for the testing set. Data after this date will be used for testing.
        :return: A tuple containing the training DataFrame and the testing DataFrame.
        """

        # Ensure test start date is part of the DataFrame
        test_start_date = pd.to_datetime(test_start_date)
        if "date" not in data.columns:
            raise ValueError("Data must contain a 'date' column for splitting.")
        if (
            test_start_date < data["date"].min()
            or test_start_date > data["date"].max()
        ):
            raise ValueError(
                f"The test start date is outside of the 'date' column values."
            )

        # Split the data into training and testing sets
        train_data = data[data["date"] < test_start_date].copy()
        test_data = data[data["date"] >= test_start_date].copy()

        train_data = self.__set_index(train_data)
        test_data = self.__set_index(test_data)

        return train_data, test_data

    def save_train_test_data(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        directory: str,
        filename: str,
    ) -> None:
        """
        Save the training and testing data to CSV files.
        :param train_data: DataFrame containing training data.
        :param test_data: DataFrame containing testing data.
        :param directory: Directory where the CSV files will be saved.
        :param filename: Base filename for the CSV files (without extension).
        """
        train_file_path = f"{directory}/{filename}_train.csv"
        test_file_path = f"{directory}/{filename}_trade.csv"

        train_data.to_csv(train_file_path, index=False)
        test_data.to_csv(test_file_path, index=False)

        print(f"Train data saved to {train_file_path}")
        print(f"Test data saved to {test_file_path}")

    def __load_file(self, filepath: str) -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame.
        :param filepath: Path to the CSV file.
        :return: DataFrame containing the data from the CSV file.
        :raises FileNotFoundError: If the file does not exist.
        :raises ValueError: If the 'tic' column is missing from the data.
        """
        try:
            data = pd.read_csv(filepath)
            # Convert date columns to datetime format if they exist
            if "date" in data.columns:
                data["date"] = pd.to_datetime(data["date"])

            # Ensure the 'tic' column exists
            if "tic" not in data.columns:
                raise ValueError("The 'tic' column is missing from the data.")

            # Set the index
            data = self.__set_index(data)

            return data

        except FileNotFoundError:
            raise FileNotFoundError(f"No file found at {filepath}")

    def load_train_test_data(
        self, directory: str, filename: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the training and testing data from CSV files.
        :param directory: Directory where the CSV files are located.
        :param filename: Base filename for the CSV files (without extension).
        :return: A tuple containing the training DataFrame and the testing DataFrame.
        """

        train_data = self.__load_file(f"{directory}/{filename}_train.csv")
        test_data = self.__load_file(f"{directory}/{filename}_trade.csv")

        return train_data, test_data
