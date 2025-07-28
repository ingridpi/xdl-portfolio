from typing import Callable

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor


class ShapExplainer:
    def __init__(self):
        """
        Initialize the SHAP explainer for the portfolio optimization environment.
        """
        pass

    def build_proxy_explainer(
        self, model: RandomForestRegressor
    ) -> shap.TreeExplainer:
        """
        Build a proxy SHAP explainer for the given model and training data.
        :param model: The trained RandomForestRegressor model.
        :param X_train: DataFrame containing the training data.
        :return: SHAP explainer object.
        """
        explainer = shap.TreeExplainer(model)
        return explainer

    def build_kernel_explainer(
        self, predict_fn: Callable, X_train: pd.DataFrame
    ) -> shap.KernelExplainer:
        """
        Build a Kernel SHAP explainer for the given prediction function and training data.
        :param predict_fn: Function to predict actions based on states.
        :param X_train: DataFrame containing the training data.
        :return: SHAP KernelExplainer object.
        """
        explainer = shap.KernelExplainer(predict_fn, X_train)
        return explainer

    def compute_shap_values(
        self, explainer: shap.Explainer, X_test: pd.DataFrame
    ) -> shap.Explanation:
        """
        Compute SHAP values for the test data using the given explainer.
        :param explainer: SHAP explainer object.
        :param X_test: DataFrame containing the test data.
        :return: SHAP explanation object containing the SHAP values.
        """
        shap_values = explainer(X_test)
        return shap_values

    def compute_shap_interaction_values(
        self, explainer: shap.Explainer, X_test: pd.DataFrame
    ) -> np.ndarray:
        """
        Compute SHAP interaction values for the test data using the given explainer.
        :param explainer: SHAP explainer object.
        :param X_test: DataFrame containing the test data.
        :return: SHAP explanation object containing the SHAP interaction values.
        """
        shap_interaction_values = explainer.shap_interaction_values(X_test)  # type: ignore
        return shap_interaction_values
