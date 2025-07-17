import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class BenchmarkVisualiser:
    def __init__(
        self,
    ):
        pass

    def compare_account_value(
        self, data: pd.DataFrame, directory: str, filename: str
    ) -> None:
        """
        Visualises the account value over time for different models.
        :param data: DataFrame containing account values with 'date', 'model', and 'account_value' columns.
        :param directory: Directory where the plot will be saved.
        :param filename: Name of the file to save the plot (without extension).
        """
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=data, x="date", y="account_value", hue="model")
        plt.title("Portfolio Value over Trading Period")
        plt.xlabel("Date")
        plt.ylabel("Account Value")
        plt.legend(title="Models")
        plt.tight_layout()
        plt.savefig(f"{directory}/{filename}_account_value.png")
        plt.show()

    def compare_daily_returns(
        self, data: pd.DataFrame, directory: str, filename: str
    ) -> None:
        """
        Visualises the daily returns over time for different models.
        :param data: DataFrame containing returns with 'date', 'model', and 'daily_return' columns.
        :param directory: Directory where the plot will be saved.
        :param filename: Name of the file to save the plot (without extension).
        """
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=data, x="date", y="daily_return", hue="model")
        plt.title("Daily Returns over Trading Period")
        plt.xlabel("Date")
        plt.ylabel("Daily Returns")
        plt.legend(title="Models")
        plt.tight_layout()
        plt.savefig(f"{directory}/{filename}_daily_returns.png")
        plt.show()

    def compare_cum_returns(
        self, data: pd.DataFrame, directory: str, filename: str
    ) -> None:
        """
        Visualises the cumulative returns over time for different models.
        :param data: DataFrame containing returns with 'date', 'model', and 'cumulative_return' columns.
        :param directory: Directory where the plot will be saved.
        :param filename: Name of the file to save the plot (without extension).
        """
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=data, x="date", y="cumulative_return", hue="model")
        plt.title("Cumulative Returns over Trading Period")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend(title="Models")
        plt.tight_layout()
        plt.savefig(f"{directory}/{filename}_cumulative_returns.png")
        plt.show()
