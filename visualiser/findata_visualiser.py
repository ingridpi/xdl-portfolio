import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class FinancialDataVisualiser:
    def __init__(self) -> None:
        """
        Initialise the FinancialDataVisualiser.
        """
        pass

    def plot_close_prices(
        self, data: pd.DataFrame, directory: str, filename: str
    ) -> None:
        """
        Plot closing prices of tickers in the data.
        :param data: DataFrame containing financial data with 'date', 'tic', and 'close' columns.
        :param directory: Directory where the plot will be saved.
        :param filename: Name of the file to save the plot (without extension).
        """
        # Sample the first 10 tickers if there are more than 10 unique tickers
        if data["tic"].nunique() > 10:
            sample_tickers = data["tic"].unique()[:10]
            data = data[data["tic"].isin(sample_tickers)]

        # Sort the data by 'tic' to ensure consistent plotting
        data.sort_values(by="tic", inplace=True)

        plt.figure(figsize=(12, 5))
        sns.lineplot(data=data, x="date", y="close", hue="tic")
        plt.title("Closing Prices of Tickers")
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.legend(title="Tickers")
        plt.tight_layout()
        plt.savefig(f"{directory}/{filename}_close_prices.png")
        plt.show()

    def plot_technical_indicators(
        self,
        data: pd.DataFrame,
        indicators: dict[str, str],
        directory: str,
        filename: str,
    ) -> None:
        """
        Plot technical indicators for each ticker in the data.
        :param data: DataFrame containing financial data with 'date', 'tic', and technical indicators.
        :param indicators: Dictionary mapping technical indicator names to their descriptions.
        :param directory: Directory where the plot will be saved.
        :param filename: Name of the file to save the plot (without extension).
        """

        # Sample the first 10 tickers if there are more than 10 unique tickers
        if data["tic"].nunique() > 10:
            sample_tickers = data["tic"].unique()[:10]
            data = data[data["tic"].isin(sample_tickers)]

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
                plt.savefig(f"{directory}/{filename}_technical.png")
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
            plt.savefig(f"{directory}/{filename}_technical.png")
            plt.show()

    def plot_macroeconomic_indicators(
        self,
        data: pd.DataFrame,
        indicators: dict[str, str],
        directory: str,
        filename: str,
    ) -> None:
        """
        Plot macroeconomic indicators.
        :param data: DataFrame containing financial data with 'date' and macroeconomic indicators.
        :param indicators: Dictionary mapping macroeconomic indicator names to their descriptions.
        :param directory: Directory where the plot will be saved.
        :param filename: Name of the file to save the plot (without extension).
        """

        # Sample the first 10 tickers if there are more than 10 unique tickers
        if data["tic"].nunique() > 10:
            sample_tickers = data["tic"].unique()[:10]
            data = data[data["tic"].isin(sample_tickers)]

        # Sort the data by 'tic' to ensure consistent plotting
        data.sort_values(by="tic", inplace=True)

        # Sample the first 5 indicators if there are more than 5 unique indicators
        ind_size = n if (n := len(indicators)) < 5 else 5

        if ind_size == 1:
            plt.figure(figsize=(12, 5))
            indicator, name = list(indicators.items())[0]
            # Convert indicator to alphanumeric
            indicator = indicator.lower().strip("^")
            if indicator in data.columns:
                # Take the date and the indicator column
                ind_df = data[["date", indicator]]
                # Remove duplicate dates
                ind_df = ind_df.drop_duplicates(subset="date")
                sns.lineplot(data=ind_df, x="date", y=indicator)
                plt.title(name)
                plt.xlabel("Date")
                plt.ylabel(indicator)
                plt.savefig(f"{directory}/{filename}_macroeconomic.png")
                plt.show()
            else:
                print(
                    f"Macroeconomic indicator '{indicator}' not found in data."
                )
        else:
            _, ax = plt.subplots(
                ind_size, 1, figsize=(12, 5 * ind_size), sharex=True
            )

            # Iterate over indicators to ind_size
            for i, (indicator, name) in enumerate(
                list(indicators.items())[:ind_size]
            ):
                # Convert indicator to alphanumeric
                indicator = indicator.lower().strip("^")
                if indicator in data.columns:
                    # Take the date and the indicator column
                    ind_df = data[["date", indicator]]
                    # Remove duplicate dates
                    ind_df = ind_df.drop_duplicates(subset="date")
                    sns.lineplot(data=ind_df, x="date", y=indicator, ax=ax[i])
                    ax[i].set_title(name)
                    ax[i].set_xlabel("Date")
                    ax[i].set_ylabel(indicator)
                    ax[i].tick_params(labelbottom=True)
                else:
                    print(
                        f"Macroeconomic indicator '{indicator}' not found in data."
                    )

            plt.tight_layout()
            plt.savefig(f"{directory}/{filename}_macroeconomic.png")
            plt.show()

    def plot_train_test_close_prices(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        directory: str,
        filename: str,
    ) -> None:
        """
        Plot closing prices for train and test datasets.
        :param train_data: DataFrame containing training data with 'date', 'tic', and 'close' columns.
        :param test_data: DataFrame containing testing data with 'date', 'tic', and 'close' columns.
        :param directory: Directory where the plot will be saved.
        :param filename: Name of the file to save the plot (without extension).
        """

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
        plt.savefig(f"{directory}/{filename}_train_test_close_prices.png")
        plt.show()
