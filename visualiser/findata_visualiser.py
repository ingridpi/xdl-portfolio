from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class FinancialDataVisualiser:
    def __init__(self, directory: str) -> None:
        """
        Initialise the FinancialDataVisualiser.
        """
        self.directory = directory

    def plot_close_prices(
        self, data: pd.DataFrame, filename: Optional[str] = "close_prices"
    ) -> None:
        """
        Plot closing prices of tickers in the data.
        :param data: DataFrame containing financial data with 'date', 'tic', and 'close' columns.
        """
        # Sample the first 10 tickers if there are more than 10 unique tickers
        if data["tic"].nunique() > 10:
            sample_tickers = data["tic"].unique()[:10]
            data = data[data["tic"].isin(sample_tickers)].copy()

        # Sort the data by 'tic' to ensure consistent plotting
        data.sort_values(by="tic", inplace=True)

        plt.figure(figsize=(12, 5))
        sns.lineplot(data=data, x="date", y="close", hue="tic")
        plt.title("Closing Prices of Tickers")
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.legend(title="Tickers")
        plt.tight_layout()
        plt.savefig(f"{self.directory}/{filename}.png")
        plt.show()

    def plot_technical_indicators(
        self,
        data: pd.DataFrame,
        indicators: dict[str, str],
    ) -> None:
        """
        Plot technical indicators for each ticker in the data.
        :param data: DataFrame containing financial data with 'date', 'tic', and technical indicators.
        :param indicators: Dictionary mapping technical indicator names to their descriptions.
        """

        # Sample the first 10 tickers if there are more than 10 unique tickers
        if data["tic"].nunique() > 10:
            sample_tickers = data["tic"].unique()[:10]
            data = data[data["tic"].isin(sample_tickers)].copy()

        # Sort the data by 'tic' to ensure consistent plotting
        data.sort_values(by="tic", inplace=True)

        # Sample the first 5 indicators if there are more than 5 unique indicators
        ind_size = n if (n := len(indicators)) < 5 else 5

        if ind_size == 1:
            plt.figure(figsize=(12, 5))
            indicator, name = list(indicators.items())[0]
            if indicator in data.columns:
                sns.lineplot(data=data, x="date", y=indicator, hue="tic")
                plt.title(name)
                plt.xlabel("Date")
                plt.ylabel(indicator)
                plt.legend(title="Tickers")
                plt.savefig(f"{self.directory}/technical_indicators.png")
                plt.show()
            else:
                print(f"Technical indicator '{indicator}' not found in data.")
        else:
            _, ax = plt.subplots(
                ind_size, 1, figsize=(12, 5 * ind_size), sharex=True
            )

            # Iterate over indicators to ind_size
            for i, (indicator, name) in enumerate(
                list(indicators.items())[:ind_size]
            ):
                if indicator in data.columns:
                    sns.lineplot(
                        data=data, x="date", y=indicator, hue="tic", ax=ax[i]
                    )
                    ax[i].set_title(name)
                    ax[i].set_xlabel("Date")
                    ax[i].set_ylabel(indicator)
                    ax[i].tick_params(labelbottom=True)
                    ax[i].legend(title="Tickers")
                else:
                    print(
                        f"Technical indicator '{indicator}' not found in data."
                    )

            plt.tight_layout()
            plt.savefig(f"{self.directory}/technical_indicators.png")
            plt.show()

    def plot_macroeconomic_indicators(
        self,
        data: pd.DataFrame,
        indicators: dict[str, str],
    ) -> None:
        """
        Plot macroeconomic indicators.
        :param data: DataFrame containing financial data with 'date' and macroeconomic indicators.
        :param indicators: Dictionary mapping macroeconomic indicator names to their descriptions.
        """

        # Sample the first 10 tickers if there are more than 10 unique tickers
        if data["tic"].nunique() > 10:
            sample_tickers = data["tic"].unique()[:10]
            data = data[data["tic"].isin(sample_tickers)].copy()

        # Sort the data by 'tic' to ensure consistent plotting
        data.sort_values(by="tic", inplace=True)

        # Sample the first 5 indicators if there are more than 5 unique indicators
        ind_size = n if (n := len(indicators)) < 5 else 5

        if ind_size == 1:
            plt.figure(figsize=(12, 5))
            indicator, name = list(indicators.items())[0]
            # Convert indicator to alphanumeric
            indicator = "".join(
                ch
                for ch in indicator.split(".", 1)[0].lower()
                if ch.isalnum() or ch == "_"
            )
            if indicator in data.columns:
                # Take the date and the indicator column
                ind_df = data[["date", indicator]]
                # Remove duplicate dates
                ind_df = ind_df.drop_duplicates(subset="date")
                sns.lineplot(data=ind_df, x="date", y=indicator)
                plt.title(name)
                plt.xlabel("Date")
                plt.ylabel(indicator)
                plt.savefig(f"{self.directory}/macroeconomic_indicators.png")
                plt.show()
            else:
                print(
                    f"Macroeconomic indicator '{indicator}' not found in data."
                )
        else:
            _, ax = plt.subplots(
                ind_size, 1, figsize=(12, 5 * ind_size), sharex=True
            )

            colors = sns.color_palette().as_hex()[
                :ind_size
            ]  # Use distinct colors

            # Iterate over indicators to ind_size
            for i, (indicator, name) in enumerate(
                list(indicators.items())[:ind_size]
            ):
                # Convert indicator to alphanumeric
                indicator = "".join(
                    ch
                    for ch in indicator.split(".", 1)[0].lower()
                    if ch.isalnum() or ch == "_"
                )
                if indicator in data.columns:
                    # Take the date and the indicator column
                    ind_df = data[["date", indicator]]
                    # Remove duplicate dates
                    ind_df = ind_df.drop_duplicates(subset="date")
                    sns.lineplot(
                        data=ind_df,
                        x="date",
                        y=indicator,
                        ax=ax[i],
                        color=colors[i],
                    )
                    ax[i].set_title(name)
                    ax[i].set_xlabel("Date")
                    ax[i].set_ylabel(indicator)
                    ax[i].tick_params(labelbottom=True)
                else:
                    print(
                        f"Macroeconomic indicator '{indicator}' not found in data."
                    )

            plt.tight_layout()
            plt.savefig(f"{self.directory}/macroeconomic_indicators.png")
            plt.show()

    def plot_train_test_close_prices(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> None:
        """
        Plot closing prices for train and test datasets.
        :param train_data: DataFrame containing training data with 'date', 'tic', and 'close' columns.
        :param test_data: DataFrame containing testing data with 'date', 'tic', and 'close' columns.
        :param directory: Directory where the plot will be saved.
        :param filename: Name of the file to save the plot (without extension).
        """

        # Sample the first 10 tickers if there are more than 10 unique tickers
        if train_data["tic"].nunique() > 10:
            sample_tickers = train_data["tic"].unique()[:10]
            train_data = train_data[
                train_data["tic"].isin(sample_tickers)
            ].copy()
            test_data = test_data[test_data["tic"].isin(sample_tickers)].copy()

        # Sort the data by 'tic' to ensure consistent plotting
        train_data.sort_values(by="tic", inplace=True)
        test_data.sort_values(by="tic", inplace=True)

        _, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True, sharey=True)

        sns.lineplot(data=train_data, x="date", y="close", hue="tic", ax=ax[0])
        ax[0].set_title("Train data set")
        ax[0].set_xlabel("Date")
        ax[0].set_ylabel("Closing Price")
        ax[0].tick_params(labelbottom=True)
        ax[0].legend(title="Tickers")

        sns.lineplot(data=test_data, x="date", y="close", hue="tic", ax=ax[1])
        ax[1].set_title("Test data set")
        ax[1].set_xlabel("Date")
        ax[1].set_ylabel("Closing Price")
        ax[1].legend(title="Tickers")

        plt.suptitle("Train and Test Closing Prices of Tickers")
        plt.tight_layout()
        plt.savefig(f"{self.directory}/train_test_close_prices.png")
        plt.show()

    def plot_train_val_test_close_prices(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> None:
        """
        Plot closing prices for train, validation, and test datasets.
        :param train_data: DataFrame containing training data with 'date', 'tic', and 'close' columns.
        :param val_data: DataFrame containing validation data with 'date', 'tic', and 'close' columns.
        :param test_data: DataFrame containing testing data with 'date', 'tic', and 'close' columns.
        """

        # Concatenate dataframes
        data = pd.concat([train_data, val_data, test_data])

        # Sort the data by 'tic' to ensure consistent plotting
        data.sort_values(by="tic", inplace=True)

        plt.figure(figsize=(12, 5))
        sns.lineplot(data=data, x="date", y="close", hue="tic")

        # Add vertical lines to indicate the split points
        plt.axvline(
            x=val_data["date"].min(),
            color="black",
            linestyle="--",
        )
        plt.axvline(x=test_data["date"].min(), color="black", linestyle="--")

        # Add labels for the split points
        plt.text(
            val_data["date"].min() + pd.Timedelta(days=20),
            data["close"].max(),
            "Validation Start",
            horizontalalignment="left",
            verticalalignment="bottom",
            color="black",
        )
        plt.text(
            test_data["date"].min() + pd.Timedelta(days=20),
            data["close"].max(),
            "Test Start",
            horizontalalignment="left",
            verticalalignment="bottom",
            color="black",
        )

        plt.title("Train, Validation, and Test Closing Prices of Tickers")
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.legend(title="Tickers")
        plt.tight_layout()
        plt.savefig(f"{self.directory}/train_val_test_close_prices.png")
        plt.show()
