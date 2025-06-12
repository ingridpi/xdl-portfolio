import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class FinancialDataVisualiser:
    def __init__(self) -> None:
        pass

    def plot_close_prices(self, data: pd.DataFrame) -> None:

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

