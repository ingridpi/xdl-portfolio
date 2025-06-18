import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class FinancialDataVisualiser:
    def __init__(self) -> None:
        pass

    def plot_close_prices(
        self, data: pd.DataFrame, directory: str, filename: str
    ) -> None:

        # Sample the first 10 tickers if there are more than 10 unique tickers
        if data.Ticker.nunique() > 10:
            sample_tickers = data["Ticker"].unique()[:10]
            data = data[data["Ticker"].isin(sample_tickers)]

        plt.figure(figsize=(12, 5))
        sns.lineplot(data=data, x="Date", y="Close", hue="Ticker")
        plt.title("Closing Prices of Tickers")
        plt.ylabel("Closing Price")
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
        """

        ind_size = n if (n := len(indicators)) < 5 else 5

        if ind_size == 1:
            plt.figure(figsize=(12, 5))
            indicator, name = list(indicators.items())[0]
            if indicator in data.columns:
                sns.lineplot(data=data, x="Date", y=indicator, hue="Ticker")
                plt.title(name)
                plt.ylabel(indicator)
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
                        data=data, x="Date", y=indicator, hue="Ticker", ax=ax[i]
                    )
                    ax[i].set_title(name)
                    ax[i].set_ylabel(indicator)
                    ax[i].tick_params(labelbottom=True)
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
        """

        ind_size = n if (n := len(indicators)) < 5 else 5

        if ind_size == 1:
            plt.figure(figsize=(12, 5))
            indicator, name = list(indicators.items())[0]
            if indicator in data.columns:
                # Take the date and the indicator column
                ind_df = data[["Date", indicator]]
                # Remove duplicate dates
                ind_df = ind_df.drop_duplicates(subset="Date")
                sns.lineplot(data=ind_df, x="Date", y=indicator)
                plt.title(name)
                plt.ylabel(indicator.strip("^"))
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
                if indicator in data.columns:
                    # Take the date and the indicator column
                    ind_df = data[["Date", indicator]]
                    # Remove duplicate dates
                    ind_df = ind_df.drop_duplicates(subset="Date")
                    sns.lineplot(data=ind_df, x="Date", y=indicator, ax=ax[i])
                    ax[i].set_title(name)
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

        _, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True, sharey=True)

        sns.lineplot(
            data=train_data, x="Date", y="Close", hue="Ticker", ax=ax[0]
        )
        ax[0].set_title("Train data set")
        ax[0].set_ylabel("Closing Price")
        ax[0].tick_params(labelbottom=True)

        sns.lineplot(
            data=test_data, x="Date", y="Close", hue="Ticker", ax=ax[1]
        )
        ax[1].set_title("Test data set")
        ax[1].set_ylabel("Closing Price")

        plt.suptitle("Train and Test Closing Prices of Tickers")
        plt.tight_layout()
        plt.savefig(f"{directory}/{filename}_train_test_close_prices.png")
        plt.show()
