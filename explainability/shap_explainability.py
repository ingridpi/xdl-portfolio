from typing import Callable, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV  # type: ignore
from sklearn.model_selection import train_test_split


class ShapExplainer:
    def __init__(self):
        """
        Initialize the SHAP explainer for the portfolio optimization environment.
        """
        self.explainer = None
        self.model = None

    def build_state_space(
        self, data: pd.DataFrame, columns: list
    ) -> pd.DataFrame:
        """
        Build the state space dataframe from a given DataFrame.
        :param data: DataFrame containing trade data with 'date' and 'tic' columns.
        :param columns: List of columns to include in the state space.
        :return: DataFrame representing the state space columns.
        """
        # Filter the trade data to include only the relevant columns
        pivot_df = data.pivot(index="date", columns="tic", values=columns)
        # Flatten the multi-index columns
        pivot_df.columns = [
            "_".join(col).strip() for col in pivot_df.columns.values
        ]

        state_space = pivot_df.reset_index()

        # Drop the 'date' column
        state_space.drop(columns=["date"], inplace=True)

        return state_space

    def build_action_space(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build the action space dataframe from a given DataFrame.
        :param data: DataFrame containing trade data with 'date' and 'tic' columns.
        :return: DataFrame representing the action space columns.
        """
        # Filter the trade data to include only the relevant columns
        action_space = data.drop(columns=["date"])

        return action_space

    def split_data(
        self,
        state_space: pd.DataFrame,
        action_space: pd.DataFrame,
        test_size: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the state and action space data into training and testing sets.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            state_space, action_space, test_size=test_size, shuffle=False
        )
        return X_train, X_test, y_train, y_test

    def build_proxy_explanation_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        find_best_params: bool = True,
    ) -> RandomForestRegressor:
        """
        Build and train a RandomForestRegressor model to explain the actions
        taken by the agent based on the state space.
        :param X_train: DataFrame containing the training state space data.
        :param y_train: DataFrame containing the training action space data.
        :param find_best_params: Whether to perform hyperparameter tuning.
        :return: The trained RandomForestRegressor model.
        """

        if find_best_params:

            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }

            grid_search = HalvingGridSearchCV(
                estimator=RandomForestRegressor(),
                param_grid=param_grid,
                cv=3,
            ).fit(X_train, y_train)
            print("Best parameters found: ", grid_search.best_params_)
            print(f"Best score: {grid_search.best_score_:.4f}")

            best_model = grid_search.best_estimator_
        else:
            best_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=4,
            )
            best_model.fit(X_train, y_train)

        return best_model

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
