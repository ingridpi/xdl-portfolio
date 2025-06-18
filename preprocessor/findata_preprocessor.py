from datetime import datetime
import itertools
import pandas as pd
from typing import List, Optional
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
        tech_indicators: Optional[List[str]] = None,
        use_macro_indicators: bool = False,
        macro_indicators: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Preprocess financial data by cleaning it and adding additional features.
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
        """
        # Placeholder for macroeconomic indicators
        # In a real implementation, this would fetch and merge actual macroeconomic data

        findownloader = FinancialDataDownloader(self.start_date, self.end_date)

        for indicator in indicators:
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
        """

        data.sort_values(by=["date", "tic"], inplace=True)

        # Set the index to a number from 0 to number of distinct dates in the dataset
        data.set_index(data["date"].astype("category").cat.codes, inplace=True)

        return data

    def split_train_test(
        self, data: pd.DataFrame, train_end_date: pd.Timestamp | str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and testing sets based on the given train end date.
        """

        # Ensure train end date is part of the DataFrame
        train_end_date = pd.to_datetime(train_end_date)
        if "date" not in data.columns:
            raise ValueError("Data must contain a 'Date' column for splitting.")
        if (
            train_end_date < data["date"].min()
            or train_end_date > data["date"].max()
        ):
            raise ValueError(
                f"The train end date is outside of the 'date' column values."
            )

        # Split the data into training and testing sets
        train_data = data[data["date"] <= train_end_date].copy()
        test_data = data[data["date"] > train_end_date].copy()

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
        """

        train_data = self.__load_file(f"{directory}/{filename}_train.csv")
        test_data = self.__load_file(f"{directory}/{filename}_trade.csv")

        return train_data, test_data
