# Backtesting and benchmarking of the trading strategies
import os
import sys

REPO_ROOT = "/Users/ingridperez/Documents/GitHub Repositiories/xdl-portfolio"
sys.path.append(REPO_ROOT)

import pandas as pd

from agents.drl_agent import DRLAgent
from config import config, config_models
from environments.env_portfolio_optimisation import (
    PortfolioOptimisationEnvWrapper,
)
from environments.env_stock_trading import StockTradingEnvWrapper
from pbenchmark.portfolio_benchmark import PortfolioBenchmark
from preprocessor.findata_preprocessor import FinancialDataPreprocessor
from visualiser.benchmark_visualiser import BenchmarkVisualiser

USE_CASE = "portfolio-optimisation"

data_dir = f"{REPO_ROOT}/{config.DATA_DIR}/{config.DATASET_NAME}"
plot_dir = f"{REPO_ROOT}/{config.PLOT_DIR}/{config.TICKERS_NAME}/{config.DATASET_NAME}/{USE_CASE}"
models_dir = f"{REPO_ROOT}/{config.MODELS_DIR}/{USE_CASE}/{config.TICKERS_NAME}/{config.DATASET_NAME}"
results_dir = (
    f"{REPO_ROOT}/{config.RESULTS_DIR}/{USE_CASE}/{config.DATASET_NAME}"
)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

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

benchmark = PortfolioBenchmark()

df_account = pd.DataFrame()
perf_stats = dict()

for model_name in config_models.MODELS.keys():

    env_train = environment.get_train_env()
    gym_env, _ = environment.get_trade_env()

    agent = DRLAgent()

    print(f"Loading model: {model_name.upper()}")
    trained_model = agent.load_model(
        model_name=model_name,
        directory=models_dir,
    )

    print(f"Evaluating model: {model_name.upper()}")
    df_account_alg, _ = agent.predict(
        model=trained_model,
        environment=gym_env,
    )

    df_account_alg["model"] = model_name.upper()

    df_account = pd.concat([df_account, df_account_alg], ignore_index=True)

    perf_stats_alg = benchmark.compute_perf_stats(df_account=df_account_alg)

    perf_stats[model_name.upper()] = perf_stats_alg

benchmark.set_data(
    train_data=train_data,
    trade_data=trade_data,
)

for strategy in ["mean", "min", "momentum", "equal"]:
    print(f"Optimising portfolio with strategy: {strategy}")
    try:
        df_account_strat = benchmark.optimise_portfolio(
            strategy=strategy,  # type: ignore
        )

        # Add cumulative returns to the account dataframe
        df_account_strat["cumulative_return"] = (
            1 + df_account_strat["daily_return"]
        ).cumprod() - 1

        df_account_strat["model"] = strategy.capitalize()
        df_account = pd.concat([df_account, df_account_strat], ignore_index=True)  # type: ignore

        perf_stats_alg = benchmark.compute_perf_stats(
            df_account=df_account_strat
        )
        perf_stats[strategy.capitalize()] = perf_stats_alg
    except Exception as e:
        print(
            f"Error occurred while optimising portfolio with strategy {strategy}: {e}"
        )

if config.INDEX is not None:
    df_account_strat = benchmark.get_index_performance(
        config.TEST_START_DATE, config.TEST_END_DATE, config.INDEX
    )

    df_account = pd.concat([df_account, df_account_strat], ignore_index=True)  # type: ignore

    perf_stats_alg = benchmark.compute_perf_stats(df_account=df_account_strat)
    perf_stats["Index"] = perf_stats_alg

perf_stats = pd.DataFrame(perf_stats)
perf_stats.to_csv(
    f"{results_dir}/{config.TICKERS_NAME}_performance_stats.csv",
    index=True,
)

benchmark_visualiser = BenchmarkVisualiser(directory=plot_dir)

benchmark_visualiser.compare_account_value(data=df_account)

benchmark_visualiser.compare_cum_returns(data=df_account)
