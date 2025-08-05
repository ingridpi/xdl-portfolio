from typing import List, Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


class FeatureImportance:
    def __init__(
        self,
        model: RandomForestRegressor,
        columns: List[str],
        directory: str,
        model_name: str,
    ):
        """
        Initialize the FeatureImportance class.
        :param model: Trained Random Forest model.
        :param columns: List of feature names.
        :param directory: Directory where plots will be saved.
        :param model_name: Name of the DRL model to which the feature importances belong to.
        """
        self.model = model
        self.columns = columns
        self.directory = directory
        self.model_name = model_name

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Calculate feature importances from the trained model.
        :return: DataFrame containing feature names and their importances.
        """
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
            .apply(lambda x: "_".join(x) if len(x) > 1 else x[0])
        )
        self.feature_importances = feature_importances

        return feature_importances

    def plot_feature_importances(
        self,
        max_features: int = 20,
    ) -> None:
        """
        Plot the feature importances
        :param max_features: Maximum number of features to display. Defaults to 20.
        :return: None
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=self.feature_importances[:max_features],
            y="feature",
            x="importance",
            orient="h",
        )
        plt.title("Feature importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(
            f"{self.directory}/{self.model_name}_feature_importance.png"
        )
        plt.show()

    def plot_feature_importances_by_ticker(
        self,
        max_features: int = 20,
    ) -> None:
        """
        Plot the feature importances by ticker.
        :return: None
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=self.feature_importances[:max_features],
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
        plt.savefig(
            f"{self.directory}/{self.model_name}_feature_importance_by_ticker.png"
        )
        plt.show()

    def add_comparison(
        self,
        statistic: Literal["mean", "median", "q1", "q3"] = "mean",
    ) -> pd.DataFrame:
        """
        Add a column to the feature importances DataFrame indicating whether
        the importance is above or below the provided statistical function.
        :param statistic: Statistical function to compare against (e.g., mean, median, q1, q3).
        :return: Updated DataFrame with a new column indicating comparison.
        """
        if statistic == "mean":
            threshold = self.feature_importances["importance"].mean()
        elif statistic == "median":
            threshold = self.feature_importances["importance"].median()
        elif statistic == "q1":
            threshold = self.feature_importances["importance"].quantile(0.25)
        elif statistic == "q3":
            threshold = self.feature_importances["importance"].quantile(0.75)
        else:
            raise ValueError(
                "Invalid statistic. Use 'mean', 'median', 'q1', or 'q3'."
            )

        self.feature_importances[statistic] = self.feature_importances[
            "importance"
        ].apply(
            lambda x: (
                f"Above {statistic}" if x > threshold else f"Below {statistic}"
            )
        )

        return self.feature_importances

    def plot_feature_importance_comparison(
        self,
        max_features: int = 20,
        statistic: Literal["mean", "median", "q1", "q3"] = "mean",
    ) -> None:
        """
        Plot the feature importances with comparison to a statistical threshold.
        :param max_features: Maximum number of features to display. Defaults to 20.
        :param statistic: Statistical function to compare against (e.g., mean, median, q1, q3).
        :return: None
        """
        feature_importances = self.add_comparison(statistic)

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
            f"{self.directory}/{self.model_name}_feature_importance_{statistic}_comparison.png"
        )
        plt.show()

    def plot_feature_importance_by_indicator(
        self,
        max_features: int = 20,
    ) -> None:
        """
        Plot the feature importances by indicator.
        :param max_features: Maximum number of features to display. Defaults to 20.
        :return: None
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=self.feature_importances[:max_features],
            y="feature",
            x="importance",
            orient="h",
            hue="indicator",
        )
        plt.title("Feature importances by Indicator")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.legend(title="Indicator")
        plt.tight_layout()
        plt.savefig(
            f"{self.directory}/{self.model_name}_feature_importance_by_indicator.png"
        )
        plt.show()

    def plot_top_features_per_ticker(
        self,
        top_features: int = 5,
    ) -> None:
        """
        Plot the top features per ticker.
        :param directory: Directory where the plot will be saved.
        :param filename: Base filename for the saved plot.
        :param top_features: Number of top features to display per ticker. Defaults to 5
        :return: None
        """

        # Group by ticker and get the top features by importance
        grouped_features = (
            self.feature_importances.groupby("ticker")[
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
        plt.savefig(
            f"{self.directory}/{self.model_name}_top_features_per_ticker.png"
        )
        plt.show()

    def plot_top_features_per_indicator(
        self,
        top_features: int = 5,
    ) -> None:
        """
        Plot the top features per indicator.
        :param top_features: Number of top features to display per indicator. Defaults to 5
        :return: None
        """

        # Select the top indicators based on their total importance
        top_indicators = (
            self.feature_importances.groupby("indicator")["importance"]
            .sum()
            .nlargest(top_features)
            .index
        )

        # Filter the feature importances to include only the top indicators
        filtered_features = self.feature_importances[
            self.feature_importances["indicator"].isin(top_indicators)
        ].copy()

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
        plt.savefig(
            f"{self.directory}/{self.model_name}_top_features_per_indicator.png"
        )
        plt.show()

    def plot_mean_importance_by_ticker(
        self,
    ) -> None:
        """
        Plot the mean feature importances by ticker.
        :return: None
        """
        mean_importances = (
            self.feature_importances.groupby("ticker")["importance"]
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
        plt.savefig(
            f"{self.directory}/{self.model_name}_mean_importance_by_ticker.png"
        )
        plt.show()

    def plot_mean_importance_by_indicator(
        self,
    ) -> None:
        """
        Plot the mean feature importances by indicator.
        :return: None
        """
        mean_importances = (
            self.feature_importances.groupby("indicator")["importance"]
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
        plt.savefig(
            f"{self.directory}/{self.model_name}_mean_importance_by_indicator.png"
        )
        plt.show()
