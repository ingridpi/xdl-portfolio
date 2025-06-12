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

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=data, x="Date", y="Close", hue="Ticker")
        plt.title("Closing Prices of Tickers")
        plt.ylabel("Closing Price")
        plt.tight_layout()
        plt.savefig(f"{directory}/{filename}_close_prices.png")
        plt.show()

    def plot_train_test_close_prices(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        directory: str,
        filename: str,
    ) -> None:

        _, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)

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
