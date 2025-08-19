# Explainability

import os
import sys

REPO_ROOT = "/Users/ingridperez/Documents/GitHub Repositiories/xdl-portfolio"
sys.path.append(REPO_ROOT)

import numpy as np
import torch

from agents.drl_agent import DRLAgent
from config import config
from environments.env_portfolio_optimisation import (
    PortfolioOptimisationEnvWrapper,
)
from environments.env_stock_trading import StockTradingEnvWrapper
from explainability.explainability import Explainer
from explainability.feature_importance import FeatureImportance
from explainability.lime_explainability import LimeExplainer
from explainability.shap_explainability import ShapExplainer
from preprocessor.findata_preprocessor import FinancialDataPreprocessor
from visualiser.shap_visualiser import ShapVisualiser

USE_CASE = "portfolio-optimisation"

data_dir = f"{REPO_ROOT}/{config.DATA_DIR}/{config.DATASET_NAME}"
plot_dir = f"{REPO_ROOT}/{config.PLOT_DIR}/{config.TICKERS_NAME}/{config.DATASET_NAME}/{USE_CASE}"
models_dir = f"{REPO_ROOT}/{config.MODELS_DIR}/{USE_CASE}/{config.TICKERS_NAME}/{config.DATASET_NAME}"
logs_dir = f"{REPO_ROOT}/{config.LOGS_DIR}/{USE_CASE}/{config.TICKERS_NAME}/{config.DATASET_NAME}"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

finpreprocessor = FinancialDataPreprocessor(
    start_date=config.START_DATE,
    end_date=config.END_DATE,
)
train_data, trade_data = finpreprocessor.load_train_test_data(
    directory=data_dir,
    filename=config.TICKERS_NAME,
)

if USE_CASE == "stock-trading":
    environment = StockTradingEnvWrapper(
        train_data=train_data,
        trade_data=trade_data,
        state_columns=config.ENVIRONMENT_COLUMNS,
    )
elif USE_CASE == "portfolio-optimisation":
    environment = PortfolioOptimisationEnvWrapper(
        train_data=train_data,
        trade_data=trade_data,
        state_columns=config.ENVIRONMENT_COLUMNS,
    )

# Load the results of a DRL agent
model_name = "a2c"

env_train = environment.get_train_env()
gym_env, _ = environment.get_trade_env()

agent = DRLAgent()

model = agent.get_model(
    model_name=model_name,
    environment=env_train,
    directory=logs_dir,
    use_case=USE_CASE,
)

print(f"Loading model: {model_name.upper()}")
trained_model = agent.load_model(
    model_name=model_name,
    directory=models_dir,
)

policy = trained_model.policy

print(f"Evaluating model: {model_name.upper()}")
df_account, _ = agent.predict(
    model=trained_model,
    environment=gym_env,
)

print(f"Evaluating model: {model_name.upper()}")
df_account, df_actions = agent.predict(trained_model, gym_env)

# Set up the explainer
explainability = Explainer()

state_space = explainability.build_state_space(
    data=trade_data,
    columns=config.ENVIRONMENT_COLUMNS,
)

action_space = explainability.build_action_space(data=df_actions)

X_train, X_test, y_train, y_test = explainability.split_data(
    state_space=state_space,
    action_space=action_space,
)

## Build Random Forest model to be used for Proxy explanations
rf_model = explainability.build_proxy_explanation_model(
    X_train=X_train,
    y_train=y_train,
    find_best_params=False,
)


## Prediction function for the policy
def predict(states: list) -> np.ndarray:
    """
    Predict the action for a given state using the policy.
    """
    with torch.no_grad():
        states_tensor = torch.tensor(states, dtype=torch.float32)
        action, _ = policy.predict(
            states_tensor.reshape(
                -1, environment.state_space, environment.stock_dim
            ).numpy()
        )
        return action


# Feature Importance

## Proxy Feature Importance
feature_imp = FeatureImportance(
    model=rf_model,
    columns=X_train.columns.to_list(),
    directory=plot_dir,
    model_name=model_name,
)
feature_importances = feature_imp.get_feature_importances()

feature_imp.plot_feature_importances()

feature_imp.plot_feature_importances_by_ticker()

feature_imp.plot_feature_importance_comparison(statistic="mean")

feature_imp.plot_feature_importance_by_indicator()

feature_imp.plot_top_features_per_ticker()

feature_imp.plot_top_features_per_indicator()

feature_imp.plot_mean_importance_by_ticker()

feature_imp.plot_mean_importance_by_indicator()

# LIME
lime_explainer = LimeExplainer(directory=plot_dir, model_name=model_name)
explainer = lime_explainer.build_lime_explainer(X_train=X_train)

## Proxy LIME
lime_explainer.explain_portfolio(
    instance=X_test.iloc[0],
    columns=action_space.columns.values.tolist(),
    predict_fn=rf_model.predict,
    filename="proxy",
)

## DRL Lime

lime_explainer.explain_portfolio(
    instance=X_test.iloc[0],
    columns=action_space.columns.values.tolist(),
    predict_fn=predict,
    filename="direct",
)

# SHAP
shap_explainer = ShapExplainer()

## Proxy SHAP
explainer = shap_explainer.build_proxy_explainer(model=rf_model)

shap_values = shap_explainer.compute_shap_values(
    explainer=explainer,
    X_test=X_test,
)

shap_interaction_values = shap_explainer.compute_shap_interaction_values(
    explainer=explainer,
    X_test=X_test,
)

shap_visualiser = ShapVisualiser(
    shap_values=shap_values,
    action_space=action_space,
    X_test=X_test,
    shap_interaction_values=shap_interaction_values,
    directory=plot_dir,
    filename="proxy",
    model_name=model_name,
)

index = 0
shap_visualiser.beeswarm_plot(index=index)

shap_visualiser.force_plot(index=index)

obs = 0
shap_visualiser.force_plot_single_obs(
    index=index,
    obs=obs,
)

shap_visualiser.waterfall_plot_single_obs(
    index=index,
    obs=obs,
)

shap_visualiser.heatmap(
    index=index,
)

shap_visualiser.interaction_plot(
    index=index,
)

## Kernel SHAP
explainer = shap_explainer.build_kernel_explainer(
    predict_fn=predict,
    X_train=X_train,
)

shap_values = shap_explainer.compute_shap_values(
    explainer=explainer,
    X_test=X_test,
)

shap_visualiser = ShapVisualiser(
    shap_values=shap_values,
    action_space=action_space,
    X_test=X_test,
    shap_interaction_values=shap_interaction_values,
    directory=plot_dir,
    filename="direct",
    model_name=model_name,
)

index = 0
shap_visualiser.beeswarm_plot(index=index)

shap_visualiser.force_plot(index=index)

obs = 0
shap_visualiser.force_plot_single_obs(
    index=index,
    obs=obs,
)

shap_visualiser.force_plot_assets(obs=obs)

shap_visualiser.waterfall_plot_single_obs(
    index=index,
    obs=obs,
)

shap_visualiser.heatmap(index=index)
