from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

shap.initjs()


class ShapVisualiser:
    def __init__(
        self,
        shap_values: shap.Explanation,
        action_space: pd.DataFrame,
        X_test: pd.DataFrame,
        directory: str,
        filename: str,
        model_name: str,
        shap_interaction_values: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initializes the SHAP visualiser with SHAP values and action space.
        :param shap_values: SHAP explanation object containing the SHAP values.
        :param action_space: DataFrame containing the action space columns.
        :param X_test: DataFrame containing the test state space data.
        :param directory: Directory where plots will be saved.
        :param filename: Base filename for the saved plots.
        :param model_name: Name of the DRL model to which the feature importances belong to.
        :param shap_interaction_values: SHAP interaction values for the features.
        """
        self.shap_values = shap_values
        self.action_space = action_space
        self.X_test = X_test
        self.directory = directory
        self.filename = filename
        self.model_name = model_name
        self.shap_interaction_values = shap_interaction_values

    def beeswarm_plot(
        self,
        index: int,
    ) -> None:
        """
        Create a beeswarm plot for the SHAP values.
        :param index: Index of the asset in the action space to plot.
        :return: None
        """
        ax = shap.plots.beeswarm(
            self.shap_values[..., index],
            show=False,
            max_display=10,
            group_remaining_features=False,
        )
        asset = self.action_space.columns[index]
        ax.set_title(f"SHAP Beeswarm Plot for {asset}")
        plt.savefig(
            f"{self.directory}/{self.model_name}_{self.filename}_shap_beeswarm_{asset}.png"
        )
        plt.show()

    def force_plot(
        self,
        index: int,
    ) -> None:
        """
        Create a force plot for the SHAP values.
        :param index: Index of the asset in the action space to plot.
        :return: None
        """
        force_plot = shap.plots.force(
            self.shap_values[..., index],
            feature_names=self.X_test.columns,
            link="logit",
        )

        shap.save_html(
            f"{self.directory}/{self.model_name}_{self.filename}_shap_force_{self.action_space.columns[index]}.html",
            force_plot,
        )

    def force_plot_single_obs(
        self,
        index: int,
        obs: int,
    ) -> None:
        """
        Create a force plot for a single observation.
        :param index: Index of the asset in the action space to plot.
        :param obs: Index of the observation to plot in time.
        :return: None
        """
        shap.plots.force(
            self.shap_values[obs, ..., index],
            feature_names=self.X_test.columns,
            link="logit",
            matplotlib=True,
            show=False,
        )
        asset = self.action_space.columns[index]

        plt.title(
            f"SHAP Force Plot for {asset} for observation {obs}",
            fontsize=16,
            y=1.5,
        )

        # Format value labels to two decimal places
        for text in plt.gca().texts:
            value = text.get_text()
            # If value contains = sign, format the number
            if "=" in value:
                parts = value.split("=")
                if len(parts) == 2:
                    try:
                        number = float(parts[1])
                        text.set_text(f"{parts[0]}= {number:.3f}")
                    except ValueError:
                        pass

        plt.savefig(
            f"{self.directory}/{self.model_name}_{self.filename}_shap_force_single_obs_{asset}_{obs}.png"
        )
        plt.show()

    def force_plot_assets(
        self,
        obs: int,
    ) -> None:
        """
        Create force plots for each asset in the portfolio.
        :param obs: Index of the observation to plot in time.
        """
        for index, _ in enumerate(self.action_space.columns):
            self.force_plot_single_obs(index, obs)

    def waterfall_plot_single_obs(
        self,
        index: int,
        obs: int,
    ) -> None:
        """
        Create a waterfall plot for the SHAP values.
        :param index: Index of the asset in the action space to plot.
        :param obs: Index of the observation to plot in time.
        """
        shap.plots.waterfall(
            self.shap_values[obs, ..., index], show=False, max_display=10
        )

        asset = self.action_space.columns[index]

        plt.title(f"SHAP Waterfall Plot for {asset}")
        plt.savefig(
            f"{self.directory}/{self.model_name}_{self.filename}_shap_waterfall_{asset}.png"
        )
        plt.show()

    def heatmap(
        self,
        index: int,
    ) -> None:
        """
        Create a heatmap for the SHAP values.
        :param index: Index of the asset in the action space to plot.
        :return: None
        """
        shap.plots.heatmap(
            self.shap_values[..., index], max_display=10, show=False
        )
        asset = self.action_space.columns[index]
        plt.title(f"SHAP Heatmap for {asset}")
        plt.savefig(
            f"{self.directory}/{self.model_name}_{self.filename}_shap_heatmap_{asset}.png"
        )
        plt.show()

    def interaction_plot(
        self,
        index: int,
    ) -> None:
        """
        Create a summary plot for the SHAP interaction values.
        :param index: Index of the asset in the action space to plot.
        """
        if self.shap_interaction_values is None:
            raise ValueError("SHAP interaction values are not provided.")

        shap.summary_plot(
            self.shap_interaction_values[..., index], self.X_test, show=False
        )

        asset = self.action_space.columns[index]
        plt.suptitle(
            f"SHAP Interaction Values for {asset}",
            y=1.1,
            fontsize=14,
        )
        plt.savefig(
            f"{self.directory}/{self.model_name}_{self.filename}_shap_interaction_{asset}.png"
        )
        plt.show()
