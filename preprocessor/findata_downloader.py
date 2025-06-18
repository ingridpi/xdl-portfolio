import pandas as pd
from typing import List
import yfinance as yf


class FinancialDataDownloader:

    def __init__(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def download_data(self, tickers: List[str]) -> pd.DataFrame:
        if self.data is not None:
            print("Data already downloaded. Returning existing data.")
            return self.data

        try:
            data = yf.download(
                tickers,
                start=self.start_date,
                end=self.end_date,
                group_by="Ticker",
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            raise Exception(f"An error occurred while downloading data: {e}")

        if data is None:
            raise ValueError(
                "No data returned from the download. Please check the tickers and date range."
            )

        # Flatten the MultiIndex columns
        data = (
            data.stack(level=0, future_stack=True)
            .rename_axis(["Date", "Ticker"])
            .reset_index(level=1)
        )

        # Reset index to have a clean DataFrame
        data.reset_index(inplace=True)

        # Rename index columns
        data.columns.rename("", inplace=True)

        # Convert column names to lowercase
        data.columns = [col.lower() for col in data.columns]

        # Rename ticker column to 'tic'
        data.rename(columns={"ticker": "tic"}, inplace=True)

        self.data = data
        print(
            f"Data downloaded for {len(tickers)} tickers from {self.start_date} to {self.end_date}."
        )
        return data

    def save_data(self, directory: str, filename: str) -> None:
        if self.data is None or self.data.empty:
            raise ValueError("No data to save. Please download data first.")

        file_path = f"{directory}/{filename}.csv"
        self.data.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")

    def load_data(self, directory: str, filename: str) -> pd.DataFrame:
        file_path = f"{directory}/{filename}.csv"

        try:
            data = pd.read_csv(file_path)

            # Convert date columns to datetime format if they exist
            data["date"] = pd.to_datetime(data["date"])

            # Sort the data by date, tic
            data.sort_values(by=["date", "tic"], inplace=True)
            # Reset index after sorting
            data.reset_index(drop=True, inplace=True)

            print(f"Data loaded from {file_path}")
            self.data = data

            return data

        except FileNotFoundError:
            raise FileNotFoundError(f"No data file found at {file_path}")

        except Exception as e:
            raise Exception(f"An error occurred while loading data: {e}")
