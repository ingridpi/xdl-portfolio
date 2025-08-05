from typing import Callable, List

import pandas as pd
from lime.lime_tabular import LimeTabularExplainer


class LimeExplainer:
    def __init__(self, directory: str, model_name: str) -> None:
        """
        Initialize the LimeExplainer with a directory to save explanation plots.
        :param directory: Directory where plots will be saved.
        :param model_name: Name of the DRL model to which the feature importances belong to.
        """
        self.directory = directory
        self.model_name = model_name

    def build_lime_explainer(
        self,
        X_train: pd.DataFrame,
    ) -> LimeTabularExplainer:
        """
        Build a LIME explainer for the given training data.
        :param X_train: DataFrame containing the training data.
        :return: LIME explainer object.
        """
        explainer = LimeTabularExplainer(
            X_train.values, mode="regression", feature_names=X_train.columns
        )
        self.explainer = explainer
        return explainer

    def explain_instance(
        self,
        instance: pd.Series,
        predict_fn: Callable,
    ) -> List:
        """
        Explain a single instance using the LIME explainer.
        Output the explanation in a notebook and return it as a list.
        :param instance: Series representing the instance to explain.
        :param predict_fn: Function to predict actions based on states.
        :return: List containing the explanation.
        """
        explanation = self.explainer.explain_instance(
            instance.values, predict_fn
        )
        explanation.show_in_notebook(show_table=True, show_all=False)
        return explanation.as_list()

    def explain_portfolio(
        self,
        instance: pd.Series,
        columns: List[str],
        predict_fn: Callable,
        filename: str,
    ) -> None:
        """
        Explain the portfolio using the LIME explainer and save the explanations as HTML files.
        :param explainer: LIME explainer object.
        :param instance: Series representing the instance to explain.
        :param columns: List of asset names to explain.
        :param predict_fn: Function to predict actions based on states.
        :param filename: Base filename for the HTML files.
        """
        # Explain each output separately
        for index, column in enumerate(columns):
            print(f"Explaining output for asset: {column}")

            # Define a predict function for the specific asset
            def predict(x):
                return predict_fn(x)[:, index]

            # Explain the instance for the specific output
            exp = self.explainer.explain_instance(
                instance.values, predict, num_features=10
            )

            # Save the explanation as HTML
            html = exp.as_html()
            with open(
                f"{self.directory}/{self.model_name}_{filename}_lime_single_obs_{column}.html",
                "w",
            ) as f:
                f.write(html)

            exp.show_in_notebook(show_table=True)
