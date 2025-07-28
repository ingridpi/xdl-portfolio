from typing import Callable, List

import pandas as pd
from lime.lime_tabular import LimeTabularExplainer


class LimeExplainer:
    def __init__(self) -> None:
        pass

    def build_lime_explainer(
        self, X_train: pd.DataFrame, feature_names: list
    ) -> LimeTabularExplainer:
        """
        Build a LIME explainer for the given training data.
        :param X_train: DataFrame containing the training data.
        :param feature_names: List of feature names.
        :return: LIME explainer object.
        """
        explainer = LimeTabularExplainer(
            X_train.values, mode="regression", feature_names=X_train.columns
        )
        return explainer

    def explain_instance(
        self,
        explainer: LimeTabularExplainer,
        instance: pd.Series,
        predict_fn: Callable,
    ) -> List:
        """
        Explain a single instance using the LIME explainer.
        :param explainer: LIME explainer object.
        :param instance: Series representing the instance to explain.
        :param predict_fn: Function to predict actions based on states.
        :return: Dictionary containing the explanation.
        """
        explanation = explainer.explain_instance(instance.values, predict_fn)
        explanation.show_in_notebook(show_table=True, show_all=False)
        return explanation.as_list()

    def explain_portfolio(
        self,
        explainer: LimeTabularExplainer,
        instance: pd.Series,
        columns: List[str],
        predict_fn: Callable,
        directory: str,
        filename: str,
    ) -> None:
        # Explain each output separately
        for index, column in enumerate(columns):
            print(f"Explaining output for asset: {column}")

            # Explain the instance for the specific output
            exp = explainer.explain_instance(
                instance.values, predict_fn, num_features=10
            )

            # Save the explanation as HTML
            html = exp.as_html()
            with open(
                f"{directory}/{filename}_lime_single_obs_{column}.html", "w"
            ) as f:
                f.write(html)

            exp.show_in_notebook(show_table=True)
