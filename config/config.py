from config import config_indicators, config_tickers

START_DATE = "2016-01-01"
END_DATE = "2025-07-01"

DATA_DIR = "data"
PLOT_DIR = "figures"
RESULTS_DIR = "results"
MODELS_DIR = "models"
LOGS_DIR = "logs"

INITIAL_AMOUNT = 100000  # Initial capital for the portfolio

TEST_NAME = "test"
DOW_30_NAME = "dow30"
EURO_STOXX_50_NAME = "eurostoxx50"
FTSE_100_NAME = "ftse100"
SP_500_NAME = "sp500"
FX_NAME = "currencies"
COMMODITY_NAME = "futures"

EXCHANGE_NYSE = "XNYS"  # New York Stock Exchange
EXCHANGE_LSE = "XLON"  # London Stock Exchange
EXCHANGE_DAX = "XFRA"  # Frankfurt Stock Exchange

TRAIN_START_DATE = START_DATE
TRAIN_END_DATE = "2022-12-31"
VAL_START_DATE = "2023-01-01"
VAL_END_DATE = "2023-12-31"
TEST_START_DATE = "2024-01-01"
TEST_END_DATE = END_DATE

# Configuration for technical indicators
USE_TECHNICAL_INDICATORS = True
USE_MACROECONOMIC_INDICATORS = True
USE_COVARIANCE_FEATURES = True

# Environment representation columns
ENVIRONMENT_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
]

if USE_TECHNICAL_INDICATORS:
    ENVIRONMENT_COLUMNS += list(config_indicators.TECHNICAL_INDICATORS.keys())

if USE_MACROECONOMIC_INDICATORS:
    MACROECONOMIC_INDICATORS = (
        config_indicators.MACROECONOMIC_INDICATORS_DEFAULT
    )

    # Remove non-letter characters from string
    ENVIRONMENT_COLUMNS += list(
        map(
            lambda x: "".join(
                ch
                for ch in x.split(".", 1)[0].lower()
                if ch.isalnum() or ch == "_"
            ),
            list(MACROECONOMIC_INDICATORS.keys()),
        )
    )

# Tickers configurations
TICKERS = config_tickers.DOW_30_TICKERS
TICKERS_NAME = DOW_30_NAME
EXCHANGE = EXCHANGE_NYSE

# Set dataset name based on the configuration
if USE_TECHNICAL_INDICATORS and USE_MACROECONOMIC_INDICATORS:
    DATASET_NAME = "dataset-indicators"
else:
    DATASET_NAME = "simple-dataset"

WANDB_ENTITY = "xdl-team"
WANDB_PROJECT = "xdl-portfolio"

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "sharpe_ratio", "goal": "maximize"},
    "early_terminate": {
        "type": "hyperband",
        "s": 2,
        "eta": 2,
        "max_iter": 10,
    },
}
