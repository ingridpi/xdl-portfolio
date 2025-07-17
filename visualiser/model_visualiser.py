from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ModelVisualiser:
    def __init__(
        self,
    ):
        pass

    def evaluate_training(
        self,
        model_name: str,
        x: str,
        y: List[str],
        title: List[str],
        logs_dir: str,
        directory: str,
        filename: str,
    ) -> None:
        """
        Visualises the training progress of an agent by plotting specified variables against a common
        x-axis variable.
        :param model_name: Name of the model being evaluated (e.g., 'a2c').
        :param x: The name of the x-axis variable (e.g., 'step').
        :param y: List of variable names to be plotted on the y-axis.
        :param title: List of titles for each subplot corresponding to the y variables.
        :param logs_dir: Directory where the training logs are stored.
        :param directory: Directory where the plot will be saved.
        :param filename: Name of the file to save the plot (without extension).
        :return: None
        """

        num_variables = len(y)
        _, ax = plt.subplots(
            num_variables, 1, figsize=(12, 5 * num_variables), sharex=True
        )

        data = pd.read_csv(f"{logs_dir}/{model_name}/progress.csv")

        colors = sns.color_palette("husl", num_variables)

        # Iterate over the variables to plot
        for i, variable in enumerate(y):
            if variable in data.columns:
                sns.lineplot(
                    data=data, x=x, y=variable, ax=ax[i], color=colors[i]
                )
                ax[i].set_title(title[i])
                ax[i].set_xlabel(x.split("/")[-1].capitalize())
                ax[i].set_ylabel(
                    " ".join(variable.split("/")[-1].split("_")).capitalize()
                )
                ax[i].tick_params(labelbottom=True)

        plt.suptitle(f"Training Progress of {model_name.upper()} Agent", y=1)
        plt.tight_layout()
        plt.savefig(f"{directory}/{filename}_{model_name}_train_evaluation.png")
        plt.show()

    def evaluate_testing(
        self,
        model_name: str,
        account_data: pd.DataFrame,
        actions_data: pd.DataFrame,
        directory: str,
        filename: str,
    ) -> None:
        """
        Visualises the testing results of an agent by plotting account value and actions over time.
        :param model_name: Name of the model being evaluated (e.g., 'a2c').
        :param account_data: DataFrame containing account values with a 'date' column.
        :param actions_data: DataFrame containing actions with a 'date' column.
        :param directory: Directory where the plot will be saved.
        :param filename: Name of the file to save the plot (without extension).
        :return: None
        """

        _, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot account value
        sns.lineplot(
            data=account_data,
            x="date",
            y="account_value",
            ax=ax[0],
        )
        ax[0].set_title("Account Value Over Time")
        ax[0].set_xlabel("Date")
        ax[0].set_ylabel("Account Value")
        ax[0].tick_params(labelbottom=True)

        # Format the actions_data
        if "date" in actions_data.columns:
            actions_data = actions_data.reset_index(drop=True).melt(
                id_vars="date", var_name="tic", value_name="action"
            )
        else:
            actions_data = actions_data.reset_index().melt(
                id_vars="date", var_name="tic", value_name="action"
            )
        actions_data.sort_values(by=["date", "tic"]).reset_index(
            drop=True, inplace=True
        )

        # Sample the first 10 tickers if there are more than 10 unique tickers
        if actions_data["tic"].nunique() > 10:
            sample_tickers = actions_data["tic"].unique()[:10]
            actions_data = actions_data[
                actions_data["tic"].isin(sample_tickers)
            ]

        # Plot actions
        sns.lineplot(
            data=actions_data,
            x="date",
            y="action",
            hue="tic",
            ax=ax[1],
        )
        ax[1].set_title("Actions Over Time")
        ax[1].set_xlabel("Date")
        ax[1].set_ylabel("Actions")
        ax[1].legend(title="Ticker")

        plt.suptitle(f"Testing Results of {model_name.upper()} Agent", y=1)
        plt.tight_layout()
        plt.savefig(f"{directory}/{filename}_{model_name}_test_evaluation.png")
        plt.show()
