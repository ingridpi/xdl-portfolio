from typing import List, Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


class FeatureImportance:
    def __init__(self, model: RandomForestRegressor, columns: List[str]):
        self.model = model
        self.columns = columns

    def get_feature_importances(self) -> pd.DataFrame:
        importances = self.model.feature_importances_
        feature_importances = pd.DataFrame(
            {
                "feature": self.columns,
                "importance": importances,
            }
        ).sort_values(by="importance", ascending=False)

        feature_importances.reset_index(drop=True, inplace=True)
        feature_importances["ticker"] = feature_importances["feature"].apply(
            lambda x: x.split("_")[-1] if "_" in x else ""
        )
        feature_importances["indicator"] = (
            feature_importances["feature"]
            .apply(lambda x: x.split("_")[:-1] if "_" in x else x)
            .apply(lambda x: "_".join(x) if isinstance(x, list) else x[0])
        )
        self.feature_importances = feature_importances

        return feature_importances

    def plot_feature_importances(
        self,
        feature_importances: pd.DataFrame,
        directory: str,
        filename: str,
        max_features: int = 20,
    ) -> None:
        """
        Plot the feature importances
        :param feature_importances: DataFrame containing feature importances.
        :param directory: Directory where the plot will be saved.
        :param filename: Base filename for the saved plot.
        :param max_features: Maximum number of features to display. Defaults to 20.
        :return: None
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=feature_importances[:max_features],
            y="feature",
            x="importance",
            orient="h",
        )
        plt.title("Feature importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(f"{directory}/{filename}_feature_importance.png")
        plt.show()

    def plot_feature_importances_by_ticker(
        self,
        feature_importances: pd.DataFrame,
        directory: str,
        filename: str,
        max_features: int = 20,
    ) -> None:
        """
        Plot the feature importances by ticker.
        :param feature_importances: DataFrame containing feature importances.
        :param directory: Directory where the plot will be saved.
        :param filename: Base filename for the saved plot.
        :return: None
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=feature_importances[:max_features],
            y="feature",
            x="importance",
            orient="h",
            hue="ticker",
        )
        plt.title("Feature importances by Ticker")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.legend(title="Ticker")
        plt.tight_layout()
        plt.savefig(f"{directory}/{filename}_feature_importance_by_ticker.png")
        plt.show()

    def add_comparison(
        self,
        feature_importances: pd.DataFrame,
        statistic: Literal["mean", "median", "q1", "q3"] = "mean",
    ) -> pd.DataFrame:
        """
        Add a column to the feature importances DataFrame indicating whether
        the importance is above or below the provided statistical function.
        :param feature_importances: DataFrame containing feature importances.
        :param statistic: Statistical function to compare against (e.g., mean, median, q1, q3).
        :return: Updated DataFrame with a new column indicating comparison.
        """
        if statistic == "mean":
            threshold = feature_importances["importance"].mean()
        elif statistic == "median":
            threshold = feature_importances["importance"].median()
        elif statistic == "q1":
            threshold = feature_importances["importance"].quantile(0.25)
        elif statistic == "q3":
            threshold = feature_importances["importance"].quantile(0.75)
        else:
            raise ValueError(
                "Invalid statistic. Use 'mean', 'median', 'q1', or 'q3'."
            )

        feature_importances[statistic] = feature_importances[
            "importance"
        ].apply(
            lambda x: (
                f"Above {statistic}" if x > threshold else f"Below {statistic}"
            )
        )
        return feature_importances

    def plot_feature_importance_comparison(
        self,
        feature_importances: pd.DataFrame,
        directory: str,
        filename: str,
        max_features: int = 20,
        statistic: Literal["mean", "median", "q1", "q3"] = "mean",
    ) -> None:
        """
        Plot the feature importances with comparison to a statistical threshold.
        :param feature_importances: DataFrame containing feature importances.
        :param directory: Directory where the plot will be saved.
        :param filename: Base filename for the saved plot.
        :param max_features: Maximum number of features to display. Defaults to 20.
        :param statistic: Statistical function to compare against (e.g., mean, median, q1, q3).
        :return: None
        """
        feature_importances = self.add_comparison(
            feature_importances, statistic
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=feature_importances[:max_features],
            y="feature",
            x="importance",
            orient="h",
            hue=statistic,
        )
        plt.title(f"Feature importances with {statistic} comparison")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.legend(title="")
        plt.tight_layout()
        plt.savefig(
            f"{directory}/{filename}_feature_importance_{statistic}_comparison.png"
        )
        plt.show()

    def plot_feature_importance_by_indicator(
        self,
        feature_importances: pd.DataFrame,
        directory: str,
        filename: str,
        max_features: int = 20,
    ) -> None:
        """
        Plot the feature importances by indicator.
        :param feature_importances: DataFrame containing feature importances.
        :param directory: Directory where the plot will be saved.
        :param filename: Base filename for the saved plot.
        :param max_features: Maximum number of features to display. Defaults to 20.
        :return: None
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=feature_importances[:max_features],
            y="feature",
            x="importance",
            orient="h",
            hue="indicator",
        )
        plt.title("Feature importances by Indicator")
        plt.xlabel("Importance")
        plt.ylabel("Indicator")
        plt.legend(title="Ticker")
        plt.tight_layout()
        plt.savefig(
            f"{directory}/{filename}_feature_importance_by_indicator.png"
        )
        plt.show()

    def plot_top_features_per_ticker(
        self,
        feature_importances: pd.DataFrame,
        directory: str,
        filename: str,
        top_features: int = 5,
    ) -> None:
        """
        Plot the top features per ticker.
        :param feature_importances: DataFrame containing feature importances.
        :param directory: Directory where the plot will be saved.
        :param filename: Base filename for the saved plot.
        :param top_features: Number of top features to display per ticker. Defaults to 5
        :return: None
        """

        # Group by ticker and get the top features by importance
        grouped_features = (
            feature_importances.groupby("ticker")[
                ["ticker", "feature", "importance"]
            ]
            .apply(lambda x: x.nlargest(top_features, "importance"))
            .reset_index(drop=True)
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=grouped_features,
            x="importance",
            y="feature",
            hue="ticker",
        )
        plt.title("Top features per Ticker")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.legend(title="Ticker")
        plt.tight_layout()
        plt.savefig(f"{directory}/{filename}_top_features_per_ticker.png")
        plt.show()

    def plot_top_features_per_indicator(
        self,
        feature_importances: pd.DataFrame,
        directory: str,
        filename: str,
        top_features: int = 5,
    ) -> None:
        """
        Plot the top features per indicator.
        :param feature_importances: DataFrame containing feature importances.
        :param directory: Directory where the plot will be saved.
        :param filename: Base filename for the saved plot.
        :param top_features: Number of top features to display per indicator. Defaults to 5
        :return: None
        """

        # Select the top indicators based on their total importance
        top_indicators = (
            feature_importances.groupby("indicator")["importance"]
            .sum()
            .nlargest(top_features)
            .index
        )

        # Filter the feature importances to include only the top indicators
        filtered_features = feature_importances[
            feature_importances["indicator"].isin(top_indicators)
        ]

        # Sort values by indicator column given the order of top_indicators
        filtered_features["indicator"] = pd.Categorical(
            filtered_features["indicator"],
            categories=top_indicators,
            ordered=True,
        )

        # Sort the filtered features by indicator and importance
        filtered_features = filtered_features.sort_values(
            by=["indicator", "importance"], ascending=[True, False]
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=filtered_features,
            x="importance",
            y="feature",
            hue="indicator",
        )
        plt.title("Top features per Indicator")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.legend(title="Indicator")
        plt.tight_layout()
        plt.savefig(f"{directory}/{filename}_top_features_per_indicator.png")
        plt.show()

    def plot_mean_importance_by_ticker(
        self,
        feature_importances: pd.DataFrame,
        directory: str,
        filename: str,
    ) -> None:
        """
        Plot the mean feature importances by ticker.
        :param feature_importances: DataFrame containing feature importances.
        :param directory: Directory where the plot will be saved.
        :param filename: Base filename for the saved plot.
        :return: None
        """
        mean_importances = (
            feature_importances.groupby("ticker")["importance"]
            .mean()
            .reset_index()
            .sort_values(by="importance", ascending=False)
        )

        plt.figure(figsize=(8, 4))
        sns.barplot(
            data=mean_importances,
            x="importance",
            y="ticker",
            orient="h",
            hue="ticker",
        )
        plt.title("Mean Feature Importances by Ticker")
        plt.xlabel("Mean Importance")
        plt.ylabel("Ticker")
        plt.tight_layout()
        plt.savefig(f"{directory}/{filename}_mean_importance_by_ticker.png")
        plt.show()

    def plot_mean_importance_by_indicator(
        self,
        feature_importances: pd.DataFrame,
        directory: str,
        filename: str,
    ) -> None:
        """
        Plot the mean feature importances by indicator.
        :param feature_importances: DataFrame containing feature importances.
        :param directory: Directory where the plot will be saved.
        :param filename: Base filename for the saved plot.
        :return: None
        """
        mean_importances = (
            feature_importances.groupby("indicator")["importance"]
            .mean()
            .reset_index()
            .sort_values(by="importance", ascending=False)
        )

        plt.figure(figsize=(8, 4))
        sns.barplot(
            data=mean_importances,
            x="importance",
            y="indicator",
            orient="h",
            hue="indicator",
        )
        plt.title("Mean Feature Importances by Indicator")
        plt.xlabel("Mean Importance")
        plt.ylabel("Indicator")
        plt.tight_layout()
        plt.savefig(f"{directory}/{filename}_mean_importance_by_indicator.png")
        plt.show()
